//! Accuracy-first temporal evaluation for recorded billiards video.
//!
//! Labels use video presentation time and source-frame identity. Detector
//! results are counted only when they were observed on that exact source
//! frame; cached state remains useful to a live client but cannot inflate
//! detection recall or freshness. Rust monotonic timestamps are deliberately
//! ignored by this offline scorer.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    decode_scene_snapshot_v1, BallStatus, CalibrationState, CueStatus, ImageSpace, Outcome,
    UpdateKind, MAX_SAFE_JSON_INTEGER,
};

pub const DEFAULT_MATCHING_RADIUS_FRACTION: f64 = 0.5;
pub const DEFAULT_FRESHNESS_BUDGET_SECONDS: f64 = 0.1;
pub const DEFAULT_MAXIMUM_ANNOTATION_BYTES: u64 = 128 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_VIDEO_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_SCENE_JSONL_BYTES: u64 = 1024 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_SCENE_LINE_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_FFPROBE_TIMEOUT_SECONDS: u64 = 30;
const ANNOTATION_SCHEMA_VERSION: u64 = 1;
const MAXIMUM_FFPROBE_STDOUT_BYTES: usize = 64 * 1024;
const MAXIMUM_FFPROBE_STDERR_BYTES: usize = 8 * 1024;

/// Annotation document schema v1, deserialized with unknown-field rejection.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationDocument {
    pub schema_version: u64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_non_null"
    )]
    pub annotation_metadata: Option<AnnotationMetadata>,
    pub sequence: AnnotationSequence,
    pub frames: Vec<AnnotationFrame>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationMetadata {
    pub source_annotation_format: String,
    pub source_annotation_sha256: String,
    pub reviewed_frames_sha256: String,
    #[serde(deserialize_with = "required_option")]
    pub epoch_events_sha256: Option<String>,
    pub frame_index_offset: i64,
    pub source_video_stream_index: u64,
    pub presentation_time_source: String,
    pub presentation_time_origin_seconds: f64,
    pub ball_radius_method: String,
    pub cue_polyline_point_order: String,
    pub tool_version: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationSequence {
    pub video_sha256: String,
    pub width: u32,
    pub height: u32,
    pub coordinate_space: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationFrame {
    pub source_frame_index: u64,
    pub presentation_time_seconds: f64,
    #[serde(deserialize_with = "required_option")]
    pub expected_calibration_revision: Option<u64>,
    #[serde(deserialize_with = "required_option")]
    pub expected_source_session_epoch: Option<String>,
    pub balls: Vec<AnnotationBall>,
    pub cue: AnnotationCue,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationBall {
    pub ball_id: String,
    pub status: AnnotationBallStatus,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_non_null"
    )]
    pub center: Option<Point2d>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_non_null"
    )]
    pub radius_px: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationBallStatus {
    Visible,
    Occluded,
    Removed,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationCue {
    pub status: AnnotationCueStatus,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_non_null"
    )]
    pub tip: Option<Point2d>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_non_null"
    )]
    pub butt: Option<Point2d>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationCueStatus {
    Visible,
    Occluded,
    Absent,
}

/// Normalized detector observation used by the engine-neutral scorer.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineFrame {
    pub source_frame_index: u64,
    pub presentation_time_seconds: f64,
    #[serde(default = "unavailable_status")]
    pub ball_positions_status: EngineStatus,
    #[serde(default)]
    pub balls: Vec<EngineBall>,
    #[serde(default)]
    pub ball_observation_source_frame_index: Option<u64>,
    #[serde(default)]
    pub ball_observation_presentation_time_seconds: Option<f64>,
    #[serde(default = "unavailable_status")]
    pub cue_observation_status: EngineStatus,
    #[serde(default)]
    pub cue_observation_source_frame_index: Option<u64>,
    #[serde(default)]
    pub cue_present: Option<bool>,
    #[serde(default)]
    pub cue_center: Option<Point2d>,
    #[serde(default)]
    pub cue_tip: Option<Point2d>,
    #[serde(default)]
    pub cue_butt: Option<Point2d>,
    #[serde(default)]
    pub cue_angle_degrees: Option<f64>,
    #[serde(default)]
    pub calibration_revision: Option<u64>,
    #[serde(default)]
    pub source_session_epoch: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineBall {
    pub center: Point2d,
    #[serde(default)]
    pub track_id: Option<TrackIdentifier>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TrackIdentifier {
    Text(String),
    Signed(i64),
    Unsigned(u64),
}

impl TrackIdentifier {
    fn as_string(&self) -> String {
        match self {
            Self::Text(value) => value.clone(),
            Self::Signed(value) => value.to_string(),
            Self::Unsigned(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EngineStatus {
    Observed,
    Cached,
    Skipped,
    Failed,
    Unavailable,
}

fn unavailable_status() -> EngineStatus {
    EngineStatus::Unavailable
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct EvaluationConfig {
    pub matching_radius_fraction: f64,
    pub freshness_budget_seconds: f64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            matching_radius_fraction: DEFAULT_MATCHING_RADIUS_FRACTION,
            freshness_budget_seconds: DEFAULT_FRESHNESS_BUDGET_SECONDS,
        }
    }
}

fn required_option<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn optional_non_null<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let value = Value::deserialize(deserializer)?;
    if value.is_null() {
        return Err(D::Error::custom("null is not allowed for this field"));
    }
    T::deserialize(value).map(Some).map_err(D::Error::custom)
}

/// Deserialize and semantically validate annotation schema v1.
pub fn parse_annotation_document(bytes: &[u8]) -> Result<AnnotationDocument> {
    let document: AnnotationDocument = serde_json::from_slice(bytes)
        .context("temporal annotation document does not match schema v1")?;
    validate_annotation_document(&document)?;
    Ok(document)
}

/// Check cross-field semantics that are not expressed by the shared JSON Schema.
pub fn validate_annotation_document(document: &AnnotationDocument) -> Result<()> {
    if document.schema_version != ANNOTATION_SCHEMA_VERSION {
        bail!("annotation schema_version must be the integer 1");
    }
    validate_sha256(&document.sequence.video_sha256, "sequence.video_sha256")?;
    if document.sequence.width == 0 || document.sequence.height == 0 {
        bail!("annotation sequence dimensions must be positive integers");
    }
    if document.sequence.coordinate_space != "source_pixels" {
        bail!("annotation sequence coordinate_space must be source_pixels");
    }
    if let Some(metadata) = &document.annotation_metadata {
        validate_metadata(metadata)?;
    }
    if document.frames.is_empty() {
        bail!("annotation frames must contain at least one frame");
    }

    let mut previous_index = None;
    let mut previous_time: Option<f64> = None;
    for (frame_number, frame) in document.frames.iter().enumerate() {
        if !frame.presentation_time_seconds.is_finite() {
            bail!("non-finite number at frames[{frame_number}].presentation_time_seconds");
        }
        if frame.presentation_time_seconds < 0.0 {
            bail!("frame presentation timestamps must be non-negative");
        }
        if previous_index.is_some_and(|previous| frame.source_frame_index <= previous) {
            bail!("annotation source_frame_index values must be strictly increasing (frame {frame_number})");
        }
        if previous_time.is_some_and(|previous| frame.presentation_time_seconds <= previous) {
            bail!("annotation presentation timestamps must be strictly increasing (frame {frame_number})");
        }
        previous_index = Some(frame.source_frame_index);
        previous_time = Some(frame.presentation_time_seconds);
        if frame
            .expected_calibration_revision
            .is_some_and(|revision| revision > MAX_SAFE_JSON_INTEGER)
        {
            bail!("expected_calibration_revision must be a non-negative safe integer or null");
        }

        let mut ball_ids = HashSet::with_capacity(frame.balls.len());
        for ball in &frame.balls {
            if ball.ball_id.is_empty() {
                bail!(
                    "ball_id must not be empty in frame {}",
                    frame.source_frame_index
                );
            }
            if !ball_ids.insert(&ball.ball_id) {
                bail!(
                    "duplicate ground-truth ball_id in frame {}",
                    frame.source_frame_index
                );
            }
            match (ball.status, ball.center, ball.radius_px) {
                (AnnotationBallStatus::Visible, Some(center), Some(radius)) => {
                    validate_point(center, "ground-truth ball center")?;
                    if !radius.is_finite() || radius <= 0.0 {
                        bail!("visible ball radius_px must be finite and greater than zero");
                    }
                }
                (AnnotationBallStatus::Visible, _, _) => {
                    bail!("visible annotation ball requires center and radius_px");
                }
                (_, None, None) => {}
                _ => bail!(
                    "occluded and removed annotation balls must not contain center or radius_px"
                ),
            }
        }
        match (frame.cue.status, frame.cue.tip, frame.cue.butt) {
            (AnnotationCueStatus::Visible, Some(tip), Some(butt)) => {
                validate_point(tip, "cue tip")?;
                validate_point(butt, "cue butt")?;
            }
            (AnnotationCueStatus::Visible, _, _) => {
                bail!("visible annotation cue requires tip and butt");
            }
            (_, None, None) => {}
            _ => bail!("occluded and absent annotation cues must not contain endpoints"),
        }
    }
    Ok(())
}

fn validate_metadata(metadata: &AnnotationMetadata) -> Result<()> {
    if metadata.source_annotation_format != "cvat_video_1.1"
        || metadata.presentation_time_source != "ffprobe_best_effort_timestamp_time"
        || metadata.ball_radius_method != "mean_box_half_extent"
        || metadata.cue_polyline_point_order != "first_tip_second_butt"
        || metadata.tool_version != 1
    {
        bail!("annotation metadata contains an unsupported v1 constant");
    }
    validate_sha256(
        &metadata.source_annotation_sha256,
        "annotation_metadata.source_annotation_sha256",
    )?;
    validate_sha256(
        &metadata.reviewed_frames_sha256,
        "annotation_metadata.reviewed_frames_sha256",
    )?;
    if let Some(hash) = &metadata.epoch_events_sha256 {
        validate_sha256(hash, "annotation_metadata.epoch_events_sha256")?;
    }
    if !metadata.presentation_time_origin_seconds.is_finite() {
        bail!("annotation metadata presentation_time_origin_seconds must be finite");
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must contain exactly 64 hexadecimal characters");
    }
    Ok(())
}

fn validate_point(point: Point2d, label: &str) -> Result<()> {
    if !point.x.is_finite() || !point.y.is_finite() {
        bail!("{label} coordinates must be finite numbers");
    }
    Ok(())
}

/// Parse normalized engine JSON into strictly ordered, finite frames.
pub fn parse_engine_frames(bytes: &[u8]) -> Result<Vec<EngineFrame>> {
    let frames: Vec<EngineFrame> =
        serde_json::from_slice(bytes).context("normalized engine frames must be a JSON array")?;
    validate_engine_frames(&frames)?;
    Ok(frames)
}

fn validate_engine_frames(frames: &[EngineFrame]) -> Result<BTreeMap<u64, EngineFrame>> {
    let mut indexed = BTreeMap::new();
    let mut previous_index = None;
    let mut previous_time: Option<f64> = None;
    for (offset, frame) in frames.iter().enumerate() {
        let index = frame.source_frame_index;
        if previous_index.is_some_and(|previous| index <= previous) {
            if indexed.contains_key(&index) {
                bail!("duplicate engine source frame {index}");
            }
            bail!("engine source frames must be strictly ordered");
        }
        if !frame.presentation_time_seconds.is_finite()
            || frame.presentation_time_seconds < 0.0
            || previous_time.is_some_and(|previous| frame.presentation_time_seconds <= previous)
        {
            bail!("engine presentation timestamps must be non-negative and strictly increasing (frame {offset})");
        }
        for ball in &frame.balls {
            validate_point(ball.center, "engine ball center")?;
        }
        if let Some(center) = frame.cue_center {
            validate_point(center, "cue_center")?;
        }
        if let Some(point) = frame.cue_tip {
            validate_point(point, "cue_tip")?;
        }
        if let Some(point) = frame.cue_butt {
            validate_point(point, "cue_butt")?;
        }
        if frame
            .cue_angle_degrees
            .is_some_and(|angle| !angle.is_finite())
        {
            bail!("cue_angle_degrees must be finite");
        }
        if frame
            .ball_observation_presentation_time_seconds
            .is_some_and(|time| !time.is_finite() || time < 0.0)
        {
            bail!("ball observation presentation time must be finite and non-negative");
        }
        if frame
            .calibration_revision
            .is_some_and(|revision| revision > MAX_SAFE_JSON_INTEGER)
        {
            bail!("engine calibration_revision must be a non-negative safe integer or null");
        }
        if frame.cue_observation_status == EngineStatus::Observed && frame.cue_present.is_none() {
            bail!("cue_present must be a boolean when cue_observation_status is observed");
        }
        if frame.cue_present == Some(true) && (frame.cue_tip.is_some() != frame.cue_butt.is_some())
        {
            bail!(
                "cue_tip and cue_butt must both be present when cue endpoint geometry is supplied"
            );
        }
        previous_index = Some(index);
        previous_time = Some(frame.presentation_time_seconds);
        indexed.insert(index, frame.clone());
    }
    Ok(indexed)
}

/// Evaluate normalized frames using exact annotation source-frame identities.
pub fn evaluate_temporal_localization(
    annotations: &AnnotationDocument,
    engine_frames: &[EngineFrame],
    config: EvaluationConfig,
) -> Result<Value> {
    if !config.matching_radius_fraction.is_finite()
        || config.matching_radius_fraction <= 0.0
        || config.matching_radius_fraction > 2.0
    {
        bail!("matching_radius_fraction must be in (0, 2]");
    }
    if !config.freshness_budget_seconds.is_finite() || config.freshness_budget_seconds < 0.0 {
        bail!("freshness_budget_seconds must be finite and non-negative");
    }
    validate_annotation_document(annotations)?;
    let engine_by_index = validate_engine_frames(engine_frames)?;
    evaluate_indexed(annotations, &engine_by_index, config)
}

fn evaluate_indexed(
    annotations: &AnnotationDocument,
    engine_by_index: &BTreeMap<u64, EngineFrame>,
    config: EvaluationConfig,
) -> Result<Value> {
    let presentation_by_index: HashMap<u64, f64> = annotations
        .frames
        .iter()
        .map(|frame| (frame.source_frame_index, frame.presentation_time_seconds))
        .collect();

    let mut visible_count = 0usize;
    let mut prediction_count = 0usize;
    let mut matched_count = 0usize;
    let mut center_errors = Vec::new();
    let mut observation_ages = Vec::new();
    let mut freshness_misses = 0usize;
    let mut stale_observation_count = 0usize;
    let mut missing_age_samples = 0usize;
    let freshness_expected_samples = annotations.frames.len();
    let mut missing_engine_frames = 0usize;
    let mut visible_starts = BTreeMap::<String, f64>::new();
    let mut first_detection = BTreeMap::<String, f64>::new();
    let mut occlusion_previous = HashMap::<String, AnnotationBallStatus>::new();
    let mut pending_reacquisitions = HashMap::<String, f64>::new();
    let mut reacquisition_delays = Vec::new();
    let mut reacquisition_attempts = 0usize;
    let mut reacquisition_successes = 0usize;

    let mut cue_tp = 0usize;
    let mut cue_fp = 0usize;
    let mut cue_fn = 0usize;
    let mut cue_tn = 0usize;
    let mut cue_labeled_count = 0usize;
    let mut cue_unavailable_engine_count = 0usize;
    let mut cue_tip_errors = Vec::new();
    let mut cue_butt_errors = Vec::new();
    let mut cue_angle_errors = Vec::new();
    let mut cue_center_errors = Vec::new();
    let mut identity_switches = 0usize;
    let mut identity_unavailable_matches = 0usize;
    let mut identified_matches = 0usize;
    let mut previous_track_id = HashMap::<String, String>::new();
    let mut previous_labeled_session: Option<String> = None;
    let mut source_session_boundary_id_resets = 0usize;

    for truth_frame in &annotations.frames {
        let source_index = truth_frame.source_frame_index;
        let current_time = truth_frame.presentation_time_seconds;
        let labeled_session = truth_frame.expected_source_session_epoch.as_deref();
        if labeled_session.is_some()
            && previous_labeled_session
                .as_deref()
                .is_some_and(|previous| Some(previous) != labeled_session)
        {
            previous_track_id.clear();
            occlusion_previous.clear();
            pending_reacquisitions.clear();
            source_session_boundary_id_resets += 1;
        }
        if let Some(session) = labeled_session {
            previous_labeled_session = Some(session.to_owned());
        }

        let result = engine_by_index.get(&source_index);
        let mut age: Option<f64> = None;
        let fresh_predictions: &[EngineBall];
        let fresh: bool;
        if let Some(result) = result {
            let observation_source_index = result.ball_observation_source_frame_index;
            let mut source_time = result.ball_observation_presentation_time_seconds;
            if source_time.is_none() {
                source_time = observation_source_index
                    .and_then(|index| presentation_by_index.get(&index).copied());
            }
            if let Some(source_time) = source_time {
                if matches!(
                    result.ball_positions_status,
                    EngineStatus::Observed | EngineStatus::Cached
                ) {
                    let computed_age = current_time - source_time;
                    if computed_age < -1e-9 {
                        bail!("engine ball observation is from the future at source frame {source_index}");
                    }
                    age = Some(computed_age.max(0.0));
                }
            }
            fresh = result.ball_positions_status == EngineStatus::Observed
                && observation_source_index == Some(source_index);
            if fresh && age.is_none() {
                age = Some(0.0);
            }
            if result.ball_positions_status == EngineStatus::Observed && !fresh {
                stale_observation_count += 1;
            }
            fresh_predictions = if fresh { &result.balls } else { &[] };
        } else {
            missing_engine_frames += 1;
            fresh_predictions = &[];
        }

        if let Some(age) = age {
            observation_ages.push(age);
            freshness_misses += usize::from(age > config.freshness_budget_seconds);
        } else {
            missing_age_samples += 1;
            freshness_misses += 1;
        }

        for truth_ball in &truth_frame.balls {
            match truth_ball.status {
                AnnotationBallStatus::Occluded => {
                    occlusion_previous
                        .insert(truth_ball.ball_id.clone(), AnnotationBallStatus::Occluded);
                }
                AnnotationBallStatus::Visible => {
                    if occlusion_previous.get(&truth_ball.ball_id)
                        == Some(&AnnotationBallStatus::Occluded)
                    {
                        pending_reacquisitions.insert(truth_ball.ball_id.clone(), current_time);
                        reacquisition_attempts += 1;
                    }
                    occlusion_previous
                        .insert(truth_ball.ball_id.clone(), AnnotationBallStatus::Visible);
                }
                AnnotationBallStatus::Removed => {
                    pending_reacquisitions.remove(&truth_ball.ball_id);
                    occlusion_previous
                        .insert(truth_ball.ball_id.clone(), AnnotationBallStatus::Removed);
                }
            }
        }

        let visible_truths: Vec<&AnnotationBall> = truth_frame
            .balls
            .iter()
            .filter(|ball| ball.status == AnnotationBallStatus::Visible)
            .collect();
        visible_count += visible_truths.len();
        prediction_count += fresh_predictions.len();
        let assignments = global_spatial_assignment(
            &visible_truths,
            fresh_predictions,
            config.matching_radius_fraction,
        )?;
        let mut matched_truth_indices = HashSet::new();
        for (truth_index, prediction_index, distance) in assignments {
            let truth_ball = visible_truths[truth_index];
            let track_key = truth_ball.ball_id.clone();
            let prediction_track_id = fresh_predictions[prediction_index]
                .track_id
                .as_ref()
                .map(TrackIdentifier::as_string);
            matched_truth_indices.insert(truth_index);
            matched_count += 1;
            center_errors.push(distance);
            visible_starts
                .entry(track_key.clone())
                .or_insert(current_time);
            first_detection
                .entry(track_key.clone())
                .or_insert(current_time);
            if let Some(track_id) = prediction_track_id {
                identified_matches += 1;
                if previous_track_id
                    .get(&track_key)
                    .is_some_and(|previous| previous != &track_id)
                {
                    identity_switches += 1;
                }
                previous_track_id.insert(track_key.clone(), track_id);
            } else {
                identity_unavailable_matches += 1;
            }
            if let Some(start_time) = pending_reacquisitions.remove(&track_key) {
                reacquisition_delays.push(current_time - start_time);
                reacquisition_successes += 1;
            }
        }

        for (truth_index, truth_ball) in visible_truths.iter().enumerate() {
            visible_starts
                .entry(truth_ball.ball_id.clone())
                .or_insert(current_time);
            // A failed visible sample leaves its reacquisition pending until a
            // later successful match or an explicit removal label.
            let _remains_pending = pending_reacquisitions.contains_key(&truth_ball.ball_id)
                && !matched_truth_indices.contains(&truth_index);
        }

        let cue_state = truth_frame.cue.status;
        let cue_result_observed = result.is_some_and(|frame| {
            frame.cue_observation_status == EngineStatus::Observed
                && frame.cue_observation_source_frame_index == Some(source_index)
        });
        if matches!(
            cue_state,
            AnnotationCueStatus::Visible | AnnotationCueStatus::Absent
        ) {
            cue_labeled_count += 1;
            if !cue_result_observed {
                cue_unavailable_engine_count += 1;
                continue;
            }
            let result = result.expect("observed cue result must have an engine frame");
            let predicted_present = result.cue_present.unwrap_or(false);
            let expected_present = cue_state == AnnotationCueStatus::Visible;
            match (expected_present, predicted_present) {
                (true, true) => {
                    cue_tp += 1;
                    let cue = &truth_frame.cue;
                    let truth_tip = cue.tip.expect("visible cue has a tip");
                    let truth_butt = cue.butt.expect("visible cue has a butt");
                    if let Some(predicted_center) = result.cue_center {
                        let truth_center = Point2d {
                            x: (truth_tip.x + truth_butt.x) / 2.0,
                            y: (truth_tip.y + truth_butt.y) / 2.0,
                        };
                        cue_center_errors.push(distance(truth_center, predicted_center));
                    }
                    if let (Some(predicted_tip), Some(predicted_butt)) =
                        (result.cue_tip, result.cue_butt)
                    {
                        cue_tip_errors.push(distance(truth_tip, predicted_tip));
                        cue_butt_errors.push(distance(truth_butt, predicted_butt));
                        let expected_angle = (truth_tip.y - truth_butt.y)
                            .atan2(truth_tip.x - truth_butt.x)
                            .to_degrees();
                        let predicted_angle = result.cue_angle_degrees.unwrap_or_else(|| {
                            (predicted_tip.y - predicted_butt.y)
                                .atan2(predicted_tip.x - predicted_butt.x)
                                .to_degrees()
                        });
                        let wrapped =
                            (predicted_angle - expected_angle + 180.0).rem_euclid(360.0) - 180.0;
                        cue_angle_errors.push(wrapped.abs());
                    }
                }
                (true, false) => cue_fn += 1,
                (false, true) => cue_fp += 1,
                (false, false) => cue_tn += 1,
            }
        }
    }

    let visible_tracks = visible_starts.len();
    let detected_tracks = first_detection.len();
    let delays: Vec<f64> = first_detection
        .iter()
        .filter_map(|(key, first)| visible_starts.get(key).map(|start| first - start))
        .collect();
    let ball_precision = ratio(matched_count, prediction_count);
    let ball_recall = ratio(matched_count, visible_count);
    let cue_precision = ratio(cue_tp, cue_tp + cue_fp);
    let cue_recall = ratio(cue_tp, cue_tp + cue_fn);
    let cue_scored = cue_tp + cue_fp + cue_fn + cue_tn;

    let mut ball_localization = json!({
        "visible_ground_truth_balls": visible_count,
        "fresh_predictions": prediction_count,
        "matched_balls": matched_count,
        "false_positives": prediction_count - matched_count,
        "false_negatives": visible_count - matched_count,
        "precision": ball_precision,
        "recall": ball_recall,
        "f1": f1(ball_precision, ball_recall),
    });
    merge_object(
        &mut ball_localization,
        summary(&center_errors, "center_error_px"),
    );

    let mut observation_freshness = json!({
        "available_samples": observation_ages.len(),
        "expected_samples": freshness_expected_samples,
        "missing_age_samples": missing_age_samples,
        "freshness_budget_seconds": config.freshness_budget_seconds,
        "freshness_budget_misses": freshness_misses,
        "freshness_budget_miss_rate": ratio(freshness_misses, freshness_expected_samples),
    });
    merge_object(
        &mut observation_freshness,
        summary(&observation_ages, "age_seconds"),
    );

    Ok(json!({
        "status": "available",
        "schema_version": 1,
        "timing_basis": "recorded-video presentation time; not camera exposure time",
        "matching": {
            "algorithm": "global minimum-cost assignment with maximum valid cardinality",
            "radius_fraction": config.matching_radius_fraction,
        },
        "availability": {
            "annotated_frames": annotations.frames.len(),
            "engine_frames": engine_by_index.len(),
            "missing_engine_frames": missing_engine_frames,
            "missing_frames_are_unavailable": true,
            "stale_observed_ball_results": stale_observation_count,
        },
        "ball_localization": ball_localization,
        "tracking": {
            "visible_tracks": visible_tracks,
            "tracks_ever_detected": detected_tracks,
            "tracks_never_detected": visible_tracks - detected_tracks,
            "identity_switches": identity_switches,
            "source_session_boundary_id_resets": source_session_boundary_id_resets,
            "matches_without_track_identity": identity_unavailable_matches,
            "identified_matches": identified_matches,
            "first_detection_delay_seconds": summary(&delays, "delay_seconds"),
            "occlusion_reacquisition_attempts": reacquisition_attempts,
            "occlusion_reacquisition_successes": reacquisition_successes,
            "occlusion_reacquisition_unresolved": reacquisition_attempts - reacquisition_successes,
            "occlusion_reacquisition_delay_seconds": summary(&reacquisition_delays, "delay_seconds"),
        },
        "observation_freshness": observation_freshness,
        "cue": {
            "labeled_presence_frames": cue_labeled_count,
            "scored_presence_frames": cue_scored,
            "unavailable_engine_frames": cue_unavailable_engine_count,
            "true_positive": cue_tp,
            "false_positive": cue_fp,
            "false_negative": cue_fn,
            "true_negative": cue_tn,
            "presence_precision": cue_precision,
            "presence_recall": cue_recall,
            "center_error_px": summary(&cue_center_errors, "error_px"),
            "tip_error_px": summary(&cue_tip_errors, "error_px"),
            "butt_error_px": summary(&cue_butt_errors, "error_px"),
            "wrapped_angle_error_degrees": summary(&cue_angle_errors, "error_degrees"),
            "occluded_frames_excluded": annotations.frames.iter().filter(|frame| frame.cue.status == AnnotationCueStatus::Occluded).count(),
        },
        "calibration": boundary_metrics(
            &annotations.frames,
            engine_by_index,
            BoundaryKind::Calibration,
        ),
        "source_session": boundary_metrics(
            &annotations.frames,
            engine_by_index,
            BoundaryKind::SourceSession,
        ),
    }))
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn f1(precision: Option<f64>, recall: Option<f64>) -> Option<f64> {
    match (precision, recall) {
        (Some(precision), Some(recall)) if precision + recall > 0.0 => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        _ => None,
    }
}

fn distance(first: Point2d, second: Point2d) -> f64 {
    (first.x - second.x).hypot(first.y - second.y)
}

fn summary(values: &[f64], prefix: &str) -> Value {
    json!({
        format!("{prefix}_count"): values.len(),
        format!("{prefix}_mean"): mean(values),
        format!("{prefix}_p50"): percentile(values, 50.0),
        format!("{prefix}_p95"): percentile(values, 95.0),
    })
}

fn merge_object(target: &mut Value, additions: Value) {
    if let (Some(target), Some(additions)) = (target.as_object_mut(), additions.as_object()) {
        target.extend(
            additions
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    // Match Python's statistics.fmean behavior. Its math.fsum-backed total
    // avoids rounding drift in audit reports when small timing values are
    // summed over large collections.
    let mut partials: Vec<f64> = Vec::new();
    for &value in values {
        let mut x = value;
        let mut retained = 0;
        let existing = partials.clone();
        for partial in existing {
            let (mut x_ordered, mut y) = (x, partial);
            if x_ordered.abs() < y.abs() {
                std::mem::swap(&mut x_ordered, &mut y);
            }
            let high = x_ordered + y;
            let low = y - (high - x_ordered);
            if low != 0.0 {
                partials[retained] = low;
                retained += 1;
            }
            x = high;
        }
        partials.truncate(retained);
        if x != 0.0 {
            partials.push(x);
        }
    }

    let mut total = partials.pop().unwrap_or(0.0);
    let mut remaining = partials.len();
    let mut low = 0.0;
    while remaining > 0 {
        remaining -= 1;
        let previous = partials[remaining];
        let high = total + previous;
        low = previous - (high - total);
        total = high;
        if low != 0.0 {
            break;
        }
    }
    if remaining > 0
        && ((low < 0.0 && partials[remaining - 1] < 0.0)
            || (low > 0.0 && partials[remaining - 1] > 0.0))
    {
        let doubled_low = low * 2.0;
        let corrected = total + doubled_low;
        if corrected - total == doubled_low {
            total = corrected;
        }
    }
    Some(total / values.len() as f64)
}

/// Match NumPy's default linear interpolation percentile method.
fn percentile(values: &[f64], percent: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * percent / 100.0;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        Some(sorted[lower])
    } else {
        let weight = position - lower as f64;
        let difference = sorted[upper] - sorted[lower];
        Some(if weight < 0.5 {
            sorted[lower] + difference * weight
        } else {
            sorted[upper] - difference * (1.0 - weight)
        })
    }
}

fn global_spatial_assignment(
    truths: &[&AnnotationBall],
    predictions: &[EngineBall],
    radius_fraction: f64,
) -> Result<Vec<(usize, usize, f64)>> {
    if truths.is_empty() || predictions.is_empty() {
        return Ok(Vec::new());
    }
    let truth_count = truths.len();
    let prediction_count = predictions.len();
    let maximum_radius = truths
        .iter()
        .filter_map(|truth| truth.radius_px)
        .fold(0.0_f64, f64::max);
    let gate_upper_bound = maximum_radius * radius_fraction;
    let unmatched_cost = (gate_upper_bound + 1.0) * (truth_count.min(prediction_count) + 1) as f64;
    let invalid_cost = unmatched_cost * (truth_count + prediction_count + 2) as f64;
    if !gate_upper_bound.is_finite() || !unmatched_cost.is_finite() || !invalid_cost.is_finite() {
        bail!("matching costs exceed the supported finite numeric range");
    }
    let mut costs = vec![vec![unmatched_cost; prediction_count + truth_count]; truth_count];
    let mut distances = vec![vec![0.0; prediction_count]; truth_count];
    for (truth_index, truth) in truths.iter().enumerate() {
        let truth_center = truth.center.expect("visible truth has a center");
        let threshold = truth.radius_px.expect("visible truth has a radius") * radius_fraction;
        for (prediction_index, prediction) in predictions.iter().enumerate() {
            let candidate_distance = distance(truth_center, prediction.center);
            if !candidate_distance.is_finite() {
                bail!("ball center distance exceeds the supported finite numeric range");
            }
            distances[truth_index][prediction_index] = candidate_distance;
            costs[truth_index][prediction_index] = if candidate_distance <= threshold {
                candidate_distance
            } else {
                invalid_cost
            };
        }
    }
    let assignment = minimum_cost_assignment(&costs)?;
    Ok(assignment
        .into_iter()
        .filter_map(|(truth_index, prediction_index)| {
            if prediction_index >= prediction_count {
                return None;
            }
            let matched_distance = distances[truth_index][prediction_index];
            (matched_distance
                <= truths[truth_index].radius_px.expect("visible radius") * radius_fraction)
                .then_some((truth_index, prediction_index, matched_distance))
        })
        .collect())
}

/// Rectangular Hungarian assignment for a finite matrix with rows <= columns.
fn minimum_cost_assignment(costs: &[Vec<f64>]) -> Result<Vec<(usize, usize)>> {
    let row_count = costs.len();
    if row_count == 0 {
        return Ok(Vec::new());
    }
    let column_count = costs[0].len();
    if column_count < row_count || costs.iter().any(|row| row.len() != column_count) {
        bail!("assignment matrix must be rectangular with at least as many columns as rows");
    }
    let mut row_potential = vec![0.0; row_count + 1];
    let mut column_potential = vec![0.0; column_count + 1];
    let mut assigned_row = vec![0usize; column_count + 1];
    let mut previous_column = vec![0usize; column_count + 1];
    for row in 1..=row_count {
        assigned_row[0] = row;
        let mut current_column = 0usize;
        let mut minimum = vec![f64::INFINITY; column_count + 1];
        let mut used = vec![false; column_count + 1];
        loop {
            used[current_column] = true;
            let current_row = assigned_row[current_column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;
            for column in 1..=column_count {
                if used[column] {
                    continue;
                }
                let reduced = costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column];
                if reduced < minimum[column] {
                    minimum[column] = reduced;
                    previous_column[column] = current_column;
                }
                if minimum[column] < delta {
                    delta = minimum[column];
                    next_column = column;
                }
            }
            if !delta.is_finite() {
                bail!("assignment could not find a finite augmenting path");
            }
            for column in 0..=column_count {
                if used[column] {
                    row_potential[assigned_row[column]] += delta;
                    column_potential[column] -= delta;
                } else {
                    minimum[column] -= delta;
                }
            }
            current_column = next_column;
            if assigned_row[current_column] == 0 {
                break;
            }
        }
        loop {
            let prior = previous_column[current_column];
            assigned_row[current_column] = assigned_row[prior];
            current_column = prior;
            if current_column == 0 {
                break;
            }
        }
    }
    let mut assignment = vec![(0usize, 0usize); row_count];
    for (column, &row) in assigned_row.iter().enumerate().skip(1) {
        if row > 0 {
            assignment[row - 1] = (row - 1, column - 1);
        }
    }
    Ok(assignment)
}

#[derive(Clone, Copy)]
enum BoundaryKind {
    Calibration,
    SourceSession,
}

fn boundary_metrics(
    annotation_frames: &[AnnotationFrame],
    engine_by_index: &BTreeMap<u64, EngineFrame>,
    kind: BoundaryKind,
) -> Value {
    let (label, expected_values, actual_values, compare_frame_values) = match kind {
        BoundaryKind::Calibration => (
            "calibration revision",
            annotation_frames
                .iter()
                .map(|frame| frame.expected_calibration_revision.map(Value::from))
                .collect::<Vec<_>>(),
            annotation_frames
                .iter()
                .map(|frame| {
                    engine_by_index
                        .get(&frame.source_frame_index)
                        .and_then(|engine| engine.calibration_revision)
                        .map(Value::from)
                })
                .collect::<Vec<_>>(),
            true,
        ),
        BoundaryKind::SourceSession => (
            "source-session epoch",
            annotation_frames
                .iter()
                .map(|frame| frame.expected_source_session_epoch.clone().map(Value::from))
                .collect::<Vec<_>>(),
            annotation_frames
                .iter()
                .map(|frame| {
                    engine_by_index
                        .get(&frame.source_frame_index)
                        .and_then(|engine| engine.source_session_epoch.clone())
                        .map(Value::from)
                })
                .collect::<Vec<_>>(),
            false,
        ),
    };
    let expected_count = expected_values
        .iter()
        .filter(|value| value.is_some())
        .count();
    let covered_count = expected_values
        .iter()
        .zip(&actual_values)
        .filter(|(expected, actual)| expected.is_some() && actual.is_some())
        .count();
    let frame_mismatches = if compare_frame_values {
        expected_values
            .iter()
            .zip(&actual_values)
            .filter(|(expected, actual)| {
                expected.is_some() && actual.is_some() && expected != actual
            })
            .count()
    } else {
        0
    };
    let mut expected_boundaries = 0usize;
    let mut actual_boundaries = 0usize;
    let mut covered_boundaries = 0usize;
    let mut boundary_mismatches = 0usize;
    for index in 1..annotation_frames.len() {
        let expected_previous = &expected_values[index - 1];
        let expected_current = &expected_values[index];
        let actual_previous = &actual_values[index - 1];
        let actual_current = &actual_values[index];
        if let (Some(expected_previous), Some(expected_current)) =
            (expected_previous, expected_current)
        {
            let expected_changed = expected_previous != expected_current;
            expected_boundaries += usize::from(expected_changed);
            if let (Some(actual_previous), Some(actual_current)) = (actual_previous, actual_current)
            {
                let actual_changed = match kind {
                    BoundaryKind::Calibration => actual_previous != actual_current,
                    BoundaryKind::SourceSession => actual_previous != actual_current,
                };
                covered_boundaries += 1;
                actual_boundaries += usize::from(actual_changed);
                boundary_mismatches += usize::from(expected_changed != actual_changed);
            }
        }
    }
    json!({
        "status": if expected_count == 0 || covered_count == 0 { "unavailable" } else if covered_count == expected_count { "complete" } else { "partial" },
        "expected_labeled_frames": expected_count,
        "covered_frames": covered_count,
        "frame_mismatch_count": if covered_count > 0 && compare_frame_values { Some(frame_mismatches) } else { None::<usize> },
        "expected_boundary_count": expected_boundaries,
        "observed_boundary_count": if covered_boundaries > 0 { Some(actual_boundaries) } else { None::<usize> },
        "covered_boundary_transitions": covered_boundaries,
        "boundary_mismatch_count": if covered_boundaries > 0 { Some(boundary_mismatches) } else { None::<usize> },
        "label": label,
    })
}

/// Normalize scene-snapshot JSONL into the engine-neutral temporal frame form.
pub fn normalize_rust_scene_jsonl(
    path: &Path,
    annotations: &AnnotationDocument,
    frame_index_offset: i64,
    maximum_total_bytes: u64,
    maximum_line_bytes: usize,
) -> Result<Vec<EngineFrame>> {
    Ok(normalize_rust_scene_file(
        path,
        annotations,
        frame_index_offset,
        maximum_total_bytes,
        maximum_line_bytes,
    )?
    .0)
}

fn normalize_rust_scene_file(
    path: &Path,
    annotations: &AnnotationDocument,
    frame_index_offset: i64,
    maximum_total_bytes: u64,
    maximum_line_bytes: usize,
) -> Result<(Vec<EngineFrame>, String)> {
    if maximum_total_bytes == 0 || maximum_line_bytes == 0 {
        bail!("scene JSONL byte limits must be positive");
    }
    validate_annotation_document(annotations)?;
    let source = File::open(path)
        .with_context(|| format!("could not read Rust scene JSONL: {}", path.display()))?;
    let mut digest = Sha256::new();
    let frames = normalize_rust_scene_reader(
        BufReader::new(source),
        annotations,
        frame_index_offset,
        maximum_total_bytes,
        maximum_line_bytes,
        &mut digest,
    )?;
    Ok((frames, format!("{:x}", digest.finalize())))
}

fn normalize_rust_scene_reader<R: BufRead>(
    mut reader: R,
    annotations: &AnnotationDocument,
    frame_index_offset: i64,
    maximum_total_bytes: u64,
    maximum_line_bytes: usize,
    digest: &mut Sha256,
) -> Result<Vec<EngineFrame>> {
    let annotation_by_index: HashMap<u64, &AnnotationFrame> = annotations
        .frames
        .iter()
        .map(|frame| (frame.source_frame_index, frame))
        .collect();
    let expected_geometry = (annotations.sequence.width, annotations.sequence.height);
    let mut normalized = Vec::new();
    let mut last_transport_sequence = BTreeMap::<(String, String), u64>::new();
    let mut seen_observation_indices = HashSet::new();
    let mut previous_observation_index = None;
    let mut total_bytes = 0u64;
    let mut line_number = 0usize;
    loop {
        let Some(line) = read_bounded_line(
            &mut reader,
            maximum_line_bytes,
            &mut total_bytes,
            maximum_total_bytes,
            digest,
        )?
        else {
            break;
        };
        line_number += 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            bail!("blank Rust scene JSONL line at {line_number}");
        }
        let snapshot = decode_scene_snapshot_v1(&line)
            .with_context(|| format!("invalid Rust scene at line {line_number}"))?;
        let identity_key = (
            snapshot.engine_instance_id.clone(),
            snapshot.source.session_id.clone(),
        );
        if last_transport_sequence
            .get(&identity_key)
            .is_some_and(|previous| snapshot.transport_sequence <= *previous)
        {
            bail!("Rust transport sequence is not increasing at line {line_number}");
        }
        last_transport_sequence.insert(identity_key, snapshot.transport_sequence);

        let one_based_index = replay_frame_index(&snapshot.source.frame_id)
            .with_context(|| format!("invalid Rust replay frame identity at line {line_number}"))?;
        let mapped_index = i128::from(one_based_index) - 1 + i128::from(frame_index_offset);
        if mapped_index < 0 || mapped_index > i128::from(u64::MAX) {
            bail!("Rust frame index maps outside the supported range at line {line_number}");
        }
        let source_index = mapped_index as u64;
        if snapshot.update_kind != UpdateKind::Observation {
            continue;
        }
        if !seen_observation_indices.insert(source_index) {
            bail!("duplicate Rust observation source frame {source_index}");
        }
        if previous_observation_index.is_some_and(|previous| source_index <= previous) {
            bail!("Rust observation frames must be strictly increasing");
        }
        previous_observation_index = Some(source_index);
        let Some(annotation_frame) = annotation_by_index.get(&source_index).copied() else {
            // Sparse labels remain sparse; never synthesize a video timestamp.
            continue;
        };
        let geometry = (snapshot.source.width_px, snapshot.source.height_px);
        if geometry != expected_geometry {
            bail!(
                "Rust scene geometry does not match temporal labels at source frame {source_index}"
            );
        }

        let ball_status = match snapshot.outcome {
            Outcome::Failed => EngineStatus::Failed,
            Outcome::Skipped => EngineStatus::Skipped,
            Outcome::Success | Outcome::Degraded => EngineStatus::Observed,
        };
        let mut balls = Vec::new();
        for ball in &snapshot.balls {
            if ball.status != BallStatus::Observed {
                continue;
            }
            let measurement = ball.measurement.as_ref().with_context(|| {
                format!("observed Rust ball has no measurement at source frame {source_index}")
            })?;
            if measurement.image_space != ImageSpace::Raw {
                bail!("Rust temporal scoring requires raw source-pixel coordinates (source frame {source_index})");
            }
            if (measurement.image_width_px, measurement.image_height_px) != expected_geometry {
                bail!("Rust ball measurement geometry differs from the labeled source image at frame {source_index}");
            }
            if ball.last_observation.session_id != snapshot.source.session_id
                || ball.last_observation.frame_id != snapshot.source.frame_id
            {
                bail!("observed Rust ball provenance does not match its scene source frame {source_index}");
            }
            balls.push(EngineBall {
                center: Point2d {
                    x: measurement.image_center_px.x,
                    y: measurement.image_center_px.y,
                },
                track_id: Some(TrackIdentifier::Text(ball.track_id.clone())),
            });
        }

        let mut cue_status = EngineStatus::Unavailable;
        let mut cue_present = None;
        let mut cue_center = None;
        if matches!(
            snapshot.cue.status,
            CueStatus::Observed | CueStatus::NotObserved
        ) && snapshot
            .cue
            .last_observation
            .as_ref()
            .is_some_and(|observation| {
                observation.session_id == snapshot.source.session_id
                    && observation.frame_id == snapshot.source.frame_id
            })
        {
            cue_status = EngineStatus::Observed;
            cue_present = Some(snapshot.cue.status == CueStatus::Observed);
            cue_center = snapshot.cue.image_center_px.map(|center| Point2d {
                x: center.x,
                y: center.y,
            });
            if cue_center.is_some()
                && (snapshot.cue.image_space != Some(ImageSpace::Raw)
                    || (snapshot.cue.image_width_px, snapshot.cue.image_height_px)
                        != (Some(expected_geometry.0), Some(expected_geometry.1)))
            {
                cue_status = EngineStatus::Unavailable;
                cue_present = None;
                cue_center = None;
            }
        }
        let calibration_revision = (snapshot.revisions.calibration_state
            == CalibrationState::Valid)
            .then_some(snapshot.revisions.calibration)
            .flatten();
        normalized.push(EngineFrame {
            source_frame_index: source_index,
            presentation_time_seconds: annotation_frame.presentation_time_seconds,
            ball_positions_status: ball_status,
            balls: if ball_status == EngineStatus::Observed {
                balls
            } else {
                Vec::new()
            },
            ball_observation_source_frame_index: (ball_status == EngineStatus::Observed)
                .then_some(source_index),
            ball_observation_presentation_time_seconds: (ball_status == EngineStatus::Observed)
                .then_some(annotation_frame.presentation_time_seconds),
            cue_observation_status: cue_status,
            cue_observation_source_frame_index: (cue_status == EngineStatus::Observed)
                .then_some(source_index),
            cue_present,
            cue_center,
            cue_tip: None,
            cue_butt: None,
            cue_angle_degrees: None,
            calibration_revision,
            source_session_epoch: Some(snapshot.source.session_id),
        });
    }
    if normalized.is_empty() {
        bail!("no Rust observation frames overlap the labeled temporal sequence");
    }
    validate_engine_frames(&normalized)?;
    Ok(normalized)
}

fn replay_frame_index(value: &str) -> Result<u64> {
    let suffix = value
        .strip_prefix("frame-")
        .context("Rust replay frame_id must use one-based frame-N identifiers")?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("Rust replay frame_id must use one-based frame-N identifiers");
    }
    let index: u64 = suffix
        .parse()
        .context("Rust replay frame index exceeds the supported integer range")?;
    if index == 0 {
        bail!("Rust replay frame_id must use one-based frame-N identifiers");
    }
    Ok(index)
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    maximum_line_bytes: usize,
    total_bytes: &mut u64,
    maximum_total_bytes: u64,
    digest: &mut Sha256,
) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let count = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if line.len().saturating_add(count) > maximum_line_bytes {
            bail!("Rust scene JSONL line exceeds the configured byte limit");
        }
        *total_bytes = total_bytes
            .checked_add(count as u64)
            .context("scene JSONL byte count overflow")?;
        if *total_bytes > maximum_total_bytes {
            bail!("Rust scene JSONL exceeds the configured total byte limit");
        }
        let ended = available[count - 1] == b'\n';
        digest.update(&available[..count]);
        line.extend_from_slice(&available[..count]);
        reader.consume(count);
        if ended {
            return Ok(Some(line));
        }
    }
}

/// Hash a file without retaining it in memory, rejecting input over the limit.
pub fn hash_file(path: &Path, maximum_bytes: u64) -> Result<(String, u64)> {
    if maximum_bytes == 0 {
        bail!("maximum file bytes must be positive");
    }
    let mut file =
        File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .context("file size overflow while hashing")?;
        if total > maximum_bytes {
            bail!("file exceeds the configured byte limit: {}", path.display());
        }
        digest.update(&buffer[..count]);
    }
    Ok((format!("{:x}", digest.finalize()), total))
}

/// Read JSON only after verifying the configured file-size bound.
pub fn read_bounded_file(path: &Path, maximum_bytes: u64, label: &str) -> Result<Vec<u8>> {
    if maximum_bytes == 0 {
        bail!("maximum {label} bytes must be positive");
    }
    let metadata = path
        .metadata()
        .with_context(|| format!("could not inspect {label}: {}", path.display()))?;
    if metadata.len() > maximum_bytes {
        bail!("{label} exceeds the configured byte limit");
    }
    let file =
        File::open(path).with_context(|| format!("could not read {label}: {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(maximum_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("could not read {label}: {}", path.display()))?;
    if bytes.len() as u64 > maximum_bytes {
        bail!("{label} exceeds the configured byte limit");
    }
    Ok(bytes)
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Query one video stream's dimensions using the configured ffprobe.
pub fn video_geometry(
    ffprobe: &Path,
    video_path: &Path,
    video_stream_index: u64,
    timeout: Duration,
) -> Result<(u32, u32)> {
    if timeout.is_zero() {
        bail!("ffprobe timeout must be positive");
    }
    let mut child = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            &format!("v:{video_stream_index}"),
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
        ])
        .arg(video_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("could not run ffprobe executable {}", ffprobe.display()))?;
    let stdout = child
        .stdout
        .take()
        .context("ffprobe stdout pipe is unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("ffprobe stderr pipe is unavailable")?;
    let stdout_reader =
        thread::spawn(move || read_capped_process_stream(stdout, MAXIMUM_FFPROBE_STDOUT_BYTES));
    let stderr_reader =
        thread::spawn(move || read_capped_process_stream(stderr, MAXIMUM_FFPROBE_STDERR_BYTES));

    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().context("could not check ffprobe status")? {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!("ffprobe exceeded its configured {timeout:?} timeout");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let (stdout, stdout_bytes) = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("ffprobe stdout reader thread panicked"))??;
    let (stderr, stderr_bytes) = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("ffprobe stderr reader thread panicked"))??;
    if stdout_bytes > MAXIMUM_FFPROBE_STDOUT_BYTES as u64 {
        bail!("ffprobe metadata exceeded the bounded output size");
    }
    if !status.success() {
        bail!(
            "ffprobe failed for {}: {}",
            video_path.display(),
            format_capped_diagnostic(&stderr, stderr_bytes)
        );
    }
    let report: Value = serde_json::from_slice(&stdout).context("ffprobe returned invalid JSON")?;
    let streams = report
        .get("streams")
        .and_then(Value::as_array)
        .context("ffprobe output has no streams array")?;
    let stream = streams
        .first()
        .context("video contains no first video stream")?;
    let width = stream
        .get("width")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .context("video has invalid frame width")?;
    let height = stream
        .get("height")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .context("video has invalid frame height")?;
    Ok((width, height))
}

fn read_capped_process_stream<R: Read>(
    mut stream: R,
    maximum_retained_bytes: usize,
) -> std::io::Result<(Vec<u8>, u64)> {
    let mut retained = Vec::with_capacity(maximum_retained_bytes.min(8 * 1024));
    let mut total_bytes = 0u64;
    let mut buffer = [0u8; 8 * 1024];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            return Ok((retained, total_bytes));
        }
        total_bytes = total_bytes.saturating_add(count as u64);
        let retained_bytes = maximum_retained_bytes
            .saturating_sub(retained.len())
            .min(count);
        retained.extend_from_slice(&buffer[..retained_bytes]);
    }
}

fn format_capped_diagnostic(bytes: &[u8], total_bytes: u64) -> String {
    let mut message = String::from_utf8_lossy(bytes).trim().to_owned();
    if total_bytes > bytes.len() as u64 {
        message.push_str(" [remaining stderr discarded]");
    }
    message
}

#[derive(Debug, Clone, Copy)]
pub struct FileEvaluationConfig {
    pub frame_index_offset: i64,
    pub evaluation: EvaluationConfig,
    pub ffprobe_timeout: Duration,
    pub maximum_video_bytes: u64,
    pub maximum_annotation_bytes: u64,
    pub maximum_scene_jsonl_bytes: u64,
    pub maximum_scene_line_bytes: usize,
}

impl Default for FileEvaluationConfig {
    fn default() -> Self {
        Self {
            frame_index_offset: 0,
            evaluation: EvaluationConfig::default(),
            ffprobe_timeout: Duration::from_secs(DEFAULT_FFPROBE_TIMEOUT_SECONDS),
            maximum_video_bytes: DEFAULT_MAXIMUM_VIDEO_BYTES,
            maximum_annotation_bytes: DEFAULT_MAXIMUM_ANNOTATION_BYTES,
            maximum_scene_jsonl_bytes: DEFAULT_MAXIMUM_SCENE_JSONL_BYTES,
            maximum_scene_line_bytes: DEFAULT_MAXIMUM_SCENE_LINE_BYTES,
        }
    }
}

/// Bind and score one exact video, annotation document, and Rust scene stream.
pub fn evaluate_files(
    video_path: &Path,
    annotation_path: &Path,
    scene_snapshot_path: &Path,
    ffprobe: &Path,
    config: FileEvaluationConfig,
) -> Result<Value> {
    for (path, label) in [
        (video_path, "video"),
        (annotation_path, "annotation file"),
        (scene_snapshot_path, "Rust scene file"),
    ] {
        if !path.is_file() {
            bail!("{label} does not exist: {}", path.display());
        }
    }
    let annotation_bytes = read_bounded_file(
        annotation_path,
        config.maximum_annotation_bytes,
        "annotation file",
    )?;
    let annotations_sha256 = hash_bytes(&annotation_bytes);
    let annotations = parse_annotation_document(&annotation_bytes)?;
    let source_video_stream_index = annotations
        .annotation_metadata
        .as_ref()
        .map(|metadata| metadata.source_video_stream_index)
        .unwrap_or(0);
    let (video_sha256, _) = hash_file(video_path, config.maximum_video_bytes)?;
    let video_geometry = video_geometry(
        ffprobe,
        video_path,
        source_video_stream_index,
        config.ffprobe_timeout,
    )?;
    if annotations.sequence.video_sha256.to_lowercase() != video_sha256 {
        bail!("annotation video SHA-256 does not match input");
    }
    if (annotations.sequence.width, annotations.sequence.height) != video_geometry {
        bail!(
            "annotation geometry does not match input video ({}x{} != {}x{})",
            annotations.sequence.width,
            annotations.sequence.height,
            video_geometry.0,
            video_geometry.1
        );
    }
    let (rust_frames, scene_snapshots_sha256) = normalize_rust_scene_file(
        scene_snapshot_path,
        &annotations,
        config.frame_index_offset,
        config.maximum_scene_jsonl_bytes,
        config.maximum_scene_line_bytes,
    )?;
    let metrics = evaluate_temporal_localization(&annotations, &rust_frames, config.evaluation)?;
    Ok(json!({
        "schema_version": 1,
        "benchmark": "rust-scene-temporal-localization",
        "metadata": {
            "video_path": absolute_path(video_path),
            "video_sha256": video_sha256,
            "video_geometry": { "width": video_geometry.0, "height": video_geometry.1 },
            "source_video_stream_index": source_video_stream_index,
            "annotations_path": absolute_path(annotation_path),
            "annotations_sha256": annotations_sha256,
            "scene_snapshots_path": absolute_path(scene_snapshot_path),
            "scene_snapshots_sha256": scene_snapshots_sha256,
            "frame_index_offset": config.frame_index_offset,
            "scored_engine_frames": rust_frames.len(),
            "timestamp_basis": "annotation video presentation time",
            "rust_monotonic_timestamps_used_for_latency": false,
            "wall_clock_output_latency_measured": false,
        },
        "temporal_localization": metrics,
    }))
}

fn absolute_path(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Explicit statement used by non-scored runs; no accuracy claim is inferred.
pub fn unavailable_temporal_metrics(reason: &str) -> Value {
    json!({
        "status": "unavailable",
        "reason": reason,
        "timing_basis": "recorded-video presentation time; not camera exposure time",
        "metrics": null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn annotation_frame(index: u64, balls: Vec<AnnotationBall>) -> AnnotationFrame {
        AnnotationFrame {
            source_frame_index: index,
            presentation_time_seconds: index as f64,
            expected_calibration_revision: Some(1),
            expected_source_session_epoch: Some("session-a".to_owned()),
            balls,
            cue: AnnotationCue {
                status: AnnotationCueStatus::Absent,
                tip: None,
                butt: None,
            },
        }
    }

    fn visible_ball(id: &str, x: f64, y: f64, radius: f64) -> AnnotationBall {
        AnnotationBall {
            ball_id: id.to_owned(),
            status: AnnotationBallStatus::Visible,
            center: Some(Point2d { x, y }),
            radius_px: Some(radius),
        }
    }

    fn annotations(frames: Vec<AnnotationFrame>) -> AnnotationDocument {
        AnnotationDocument {
            schema_version: 1,
            annotation_metadata: None,
            sequence: AnnotationSequence {
                video_sha256: "a".repeat(64),
                width: 100,
                height: 80,
                coordinate_space: "source_pixels".to_owned(),
            },
            frames,
        }
    }

    fn engine_frame(index: u64, balls: Vec<EngineBall>, status: EngineStatus) -> EngineFrame {
        EngineFrame {
            source_frame_index: index,
            presentation_time_seconds: index as f64,
            ball_positions_status: status,
            balls,
            ball_observation_source_frame_index: (status == EngineStatus::Observed)
                .then_some(index),
            ball_observation_presentation_time_seconds: (status == EngineStatus::Observed)
                .then_some(index as f64),
            cue_observation_status: EngineStatus::Unavailable,
            cue_observation_source_frame_index: None,
            cue_present: None,
            cue_center: None,
            cue_tip: None,
            cue_butt: None,
            cue_angle_degrees: None,
            calibration_revision: Some(1),
            source_session_epoch: Some("session-a".to_owned()),
        }
    }

    #[test]
    fn annotation_schema_rejects_duplicate_ids_order_and_nonfinite_values() {
        let valid = AnnotationDocument {
            schema_version: 1,
            annotation_metadata: None,
            sequence: AnnotationSequence {
                video_sha256: "A".repeat(64),
                width: 100,
                height: 80,
                coordinate_space: "source_pixels".to_owned(),
            },
            frames: vec![annotation_frame(0, vec![visible_ball("a", 3.0, 4.0, 5.0)])],
        };
        validate_annotation_document(&valid).unwrap();

        let mut duplicate = valid.clone();
        let duplicate_ball = duplicate.frames[0].balls[0].clone();
        duplicate.frames[0].balls.push(duplicate_ball);
        assert!(validate_annotation_document(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate ground-truth"));

        let mut nonincreasing = valid.clone();
        nonincreasing.frames.push(nonincreasing.frames[0].clone());
        assert!(validate_annotation_document(&nonincreasing).is_err());

        let mut nonfinite = valid;
        nonfinite.frames[0].balls[0].center.as_mut().unwrap().x = f64::INFINITY;
        assert!(validate_annotation_document(&nonfinite).is_err());
    }

    #[test]
    fn strict_annotation_parser_requires_nullable_keys_and_rejects_unknown_fields() {
        let document = annotations(vec![annotation_frame(0, Vec::new())]);
        let encoded = serde_json::to_vec(&document).unwrap();
        parse_annotation_document(&encoded).unwrap();

        let mut value: Value = serde_json::from_slice(&encoded).unwrap();
        value["frames"][0]
            .as_object_mut()
            .unwrap()
            .remove("expected_calibration_revision");
        assert!(parse_annotation_document(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut value: Value = serde_json::from_slice(&encoded).unwrap();
        value["extra"] = json!(true);
        assert!(parse_annotation_document(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn global_assignment_maximizes_valid_matches_and_cached_frames_are_not_detections() {
        let annotations = annotations(vec![annotation_frame(
            0,
            vec![
                visible_ball("near-left", 0.0, 0.0, 10.0),
                visible_ball("near-right", 4.0, 0.0, 10.0),
            ],
        )]);
        let engine = vec![engine_frame(
            0,
            vec![
                EngineBall {
                    center: Point2d { x: 1.0, y: 0.0 },
                    track_id: Some(TrackIdentifier::Text("one".to_owned())),
                },
                EngineBall {
                    center: Point2d { x: -4.0, y: 0.0 },
                    track_id: Some(TrackIdentifier::Text("two".to_owned())),
                },
            ],
            EngineStatus::Observed,
        )];
        let report =
            evaluate_temporal_localization(&annotations, &engine, EvaluationConfig::default())
                .unwrap();
        assert_eq!(report["ball_localization"]["matched_balls"], 2);
        assert_eq!(report["ball_localization"]["false_negatives"], 0);

        let mut cached = engine[0].clone();
        cached.ball_positions_status = EngineStatus::Cached;
        let report =
            evaluate_temporal_localization(&annotations, &[cached], EvaluationConfig::default())
                .unwrap();
        assert_eq!(report["ball_localization"]["fresh_predictions"], 0);
        assert_eq!(report["ball_localization"]["recall"], 0.0);
    }

    #[test]
    fn global_assignment_handles_unmatched_truths_and_out_of_gate_predictions() {
        let annotations = annotations(vec![annotation_frame(
            0,
            vec![
                visible_ball("matched", 0.0, 0.0, 10.0),
                visible_ball("unmatched", 100.0, 100.0, 10.0),
            ],
        )]);
        let engine = vec![engine_frame(
            0,
            vec![EngineBall {
                center: Point2d { x: 0.0, y: 0.0 },
                track_id: None,
            }],
            EngineStatus::Observed,
        )];

        let report =
            evaluate_temporal_localization(&annotations, &engine, EvaluationConfig::default())
                .unwrap();

        assert_eq!(report["ball_localization"]["matched_balls"], 1);
        assert_eq!(report["ball_localization"]["false_negatives"], 1);
        assert_eq!(report["ball_localization"]["false_positives"], 0);
    }

    #[test]
    fn cached_missing_and_stale_results_remain_distinct() {
        let annotation_document = annotations(vec![
            annotation_frame(0, vec![visible_ball("ball", 10.0, 10.0, 8.0)]),
            annotation_frame(1, vec![visible_ball("ball", 10.0, 10.0, 8.0)]),
        ]);
        let first = engine_frame(
            0,
            vec![EngineBall {
                center: Point2d { x: 10.0, y: 10.0 },
                track_id: Some(TrackIdentifier::Text("track".to_owned())),
            }],
            EngineStatus::Observed,
        );
        let mut cached = first.clone();
        cached.source_frame_index = 1;
        cached.presentation_time_seconds = 1.0;
        cached.ball_positions_status = EngineStatus::Cached;
        cached.ball_observation_presentation_time_seconds = Some(0.0);

        let report = evaluate_temporal_localization(
            &annotation_document,
            &[first.clone(), cached],
            EvaluationConfig {
                freshness_budget_seconds: 0.25,
                ..EvaluationConfig::default()
            },
        )
        .unwrap();
        assert_eq!(report["ball_localization"]["fresh_predictions"], 1);
        assert_eq!(report["ball_localization"]["false_negatives"], 1);
        assert_eq!(report["observation_freshness"]["age_seconds_mean"], 0.5);
        assert_eq!(
            report["observation_freshness"]["freshness_budget_misses"],
            1
        );

        let missing = evaluate_temporal_localization(
            &annotation_document,
            std::slice::from_ref(&first),
            EvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(missing["availability"]["missing_engine_frames"], 1);
        assert_eq!(missing["observation_freshness"]["missing_age_samples"], 1);
        assert_eq!(
            missing["observation_freshness"]["freshness_budget_misses"],
            1
        );

        let mut stale = first;
        stale.ball_observation_source_frame_index = Some(4);
        let stale_report = evaluate_temporal_localization(
            &annotations(vec![annotation_frame(
                0,
                vec![visible_ball("ball", 10.0, 10.0, 8.0)],
            )]),
            &[stale],
            EvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(
            stale_report["availability"]["stale_observed_ball_results"],
            1
        );
        assert_eq!(stale_report["ball_localization"]["fresh_predictions"], 0);
        assert_eq!(stale_report["ball_localization"]["false_negatives"], 1);
    }

    #[test]
    fn cue_presence_endpoints_and_wrapped_angle_are_scored_separately() {
        let visible_cue = AnnotationCue {
            status: AnnotationCueStatus::Visible,
            tip: Some(Point2d {
                x: -1.0,
                y: 0.1763269807,
            }),
            butt: Some(Point2d { x: 0.0, y: 0.0 }),
        };
        let occluded_cue = AnnotationCue {
            status: AnnotationCueStatus::Occluded,
            tip: None,
            butt: None,
        };
        let mut annotations = annotations(vec![
            annotation_frame(0, Vec::new()),
            annotation_frame(1, Vec::new()),
            annotation_frame(2, Vec::new()),
        ]);
        annotations.frames[0].cue = visible_cue;
        annotations.frames[1].cue.status = AnnotationCueStatus::Absent;
        annotations.frames[2].cue = occluded_cue;

        let mut visible = engine_frame(0, Vec::new(), EngineStatus::Observed);
        visible.cue_observation_status = EngineStatus::Observed;
        visible.cue_observation_source_frame_index = Some(0);
        visible.cue_present = Some(true);
        visible.cue_tip = Some(Point2d {
            x: 0.0,
            y: 0.1763269807,
        });
        visible.cue_butt = Some(Point2d { x: 0.0, y: 0.0 });
        visible.cue_angle_degrees = Some(-170.0);

        let mut false_positive = engine_frame(1, Vec::new(), EngineStatus::Observed);
        false_positive.cue_observation_status = EngineStatus::Observed;
        false_positive.cue_observation_source_frame_index = Some(1);
        false_positive.cue_present = Some(true);

        let mut unavailable = engine_frame(2, Vec::new(), EngineStatus::Observed);
        unavailable.cue_observation_status = EngineStatus::Skipped;
        unavailable.cue_observation_source_frame_index = None;
        unavailable.cue_present = None;

        let report = evaluate_temporal_localization(
            &annotations,
            &[visible, false_positive, unavailable],
            EvaluationConfig::default(),
        )
        .unwrap();
        let cue = &report["cue"];
        assert_eq!(cue["presence_precision"], 0.5);
        assert_eq!(cue["presence_recall"], 1.0);
        assert_eq!(cue["tip_error_px"]["error_px_mean"], 1.0);
        assert_eq!(cue["butt_error_px"]["error_px_mean"], 0.0);
        let angle_error = cue["wrapped_angle_error_degrees"]["error_degrees_mean"]
            .as_f64()
            .unwrap();
        assert!((angle_error - 20.0).abs() < 1e-6);
        assert_eq!(cue["occluded_frames_excluded"], 1);
    }

    #[test]
    fn calibration_and_session_boundaries_have_independent_coverage() {
        let annotations = annotations(vec![
            {
                let mut frame = annotation_frame(0, Vec::new());
                frame.expected_calibration_revision = Some(0);
                frame.expected_source_session_epoch = Some("session-a".to_owned());
                frame
            },
            {
                let mut frame = annotation_frame(1, Vec::new());
                frame.expected_calibration_revision = Some(0);
                frame.expected_source_session_epoch = Some("session-a".to_owned());
                frame
            },
            {
                let mut frame = annotation_frame(2, Vec::new());
                frame.expected_calibration_revision = Some(1);
                frame.expected_source_session_epoch = Some("session-b".to_owned());
                frame
            },
        ]);
        let engine = vec![
            {
                let mut frame = engine_frame(0, Vec::new(), EngineStatus::Observed);
                frame.calibration_revision = Some(0);
                frame.source_session_epoch = Some("rust-session-x".to_owned());
                frame
            },
            {
                let mut frame = engine_frame(1, Vec::new(), EngineStatus::Observed);
                frame.calibration_revision = Some(1);
                frame.source_session_epoch = Some("rust-session-x".to_owned());
                frame
            },
            {
                let mut frame = engine_frame(2, Vec::new(), EngineStatus::Observed);
                frame.calibration_revision = Some(1);
                frame.source_session_epoch = Some("rust-session-x".to_owned());
                frame
            },
        ];

        let report =
            evaluate_temporal_localization(&annotations, &engine, EvaluationConfig::default())
                .unwrap();
        assert_eq!(report["calibration"]["frame_mismatch_count"], 1);
        assert_eq!(report["calibration"]["expected_boundary_count"], 1);
        assert_eq!(report["calibration"]["boundary_mismatch_count"], 2);
        assert_eq!(
            report["source_session"]["frame_mismatch_count"],
            Value::Null
        );
        assert_eq!(report["source_session"]["expected_boundary_count"], 1);
        assert_eq!(report["source_session"]["boundary_mismatch_count"], 1);

        let mut without_sessions = engine;
        for frame in &mut without_sessions {
            frame.source_session_epoch = None;
        }
        let report = evaluate_temporal_localization(
            &annotations,
            &without_sessions,
            EvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(report["source_session"]["status"], "unavailable");
        assert_eq!(report["source_session"]["expected_boundary_count"], 1);
        assert_eq!(
            report["source_session"]["frame_mismatch_count"],
            Value::Null
        );
    }

    #[test]
    fn missing_frames_count_as_unavailable_freshness_and_reacquisition_resets_session_ids() {
        let mut middle = annotation_frame(1, vec![visible_ball("ball", 2.0, 2.0, 4.0)]);
        middle.balls[0].status = AnnotationBallStatus::Occluded;
        middle.balls[0].center = None;
        middle.balls[0].radius_px = None;
        let annotations = annotations(vec![
            annotation_frame(0, vec![visible_ball("ball", 2.0, 2.0, 4.0)]),
            middle,
            annotation_frame(2, vec![visible_ball("ball", 2.0, 2.0, 4.0)]),
            {
                let mut frame = annotation_frame(3, vec![visible_ball("ball", 2.0, 2.0, 4.0)]);
                frame.expected_source_session_epoch = Some("session-b".to_owned());
                frame.expected_calibration_revision = Some(2);
                frame
            },
        ]);
        let engine = vec![
            engine_frame(
                0,
                vec![EngineBall {
                    center: Point2d { x: 2.0, y: 2.0 },
                    track_id: Some(TrackIdentifier::Text("old".to_owned())),
                }],
                EngineStatus::Observed,
            ),
            engine_frame(
                2,
                vec![EngineBall {
                    center: Point2d { x: 2.0, y: 2.0 },
                    track_id: Some(TrackIdentifier::Text("new".to_owned())),
                }],
                EngineStatus::Observed,
            ),
            {
                let mut frame = engine_frame(
                    3,
                    vec![EngineBall {
                        center: Point2d { x: 2.0, y: 2.0 },
                        track_id: Some(TrackIdentifier::Text("new".to_owned())),
                    }],
                    EngineStatus::Observed,
                );
                frame.source_session_epoch = Some("session-b".to_owned());
                frame.calibration_revision = Some(2);
                frame
            },
        ];
        let report =
            evaluate_temporal_localization(&annotations, &engine, EvaluationConfig::default())
                .unwrap();
        assert_eq!(report["observation_freshness"]["missing_age_samples"], 1);
        assert_eq!(report["tracking"]["source_session_boundary_id_resets"], 1);
        assert_eq!(report["tracking"]["identity_switches"], 1);
        assert_eq!(report["tracking"]["occlusion_reacquisition_successes"], 1);
        assert_eq!(report["calibration"]["frame_mismatch_count"], 0);
        assert_eq!(report["source_session"]["boundary_mismatch_count"], 0);
    }

    #[test]
    fn percentile_matches_numpy_linear_interpolation() {
        assert_eq!(percentile(&[0.0, 10.0, 20.0, 30.0], 50.0), Some(15.0));
        assert_eq!(percentile(&[0.0, 10.0, 20.0, 30.0], 95.0), Some(28.5));
        assert_eq!(percentile(&[], 95.0), None);
        assert_eq!(
            percentile(&[4.089576548481155, 37.90196043954357], 95.0),
            Some(36.21134124499045)
        );
    }

    #[test]
    fn shared_temporal_evaluation_fixture_matches_the_python_reference_report() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../contracts/v1/temporal-evaluation-golden.json"
        ))
        .unwrap();
        let annotations =
            parse_annotation_document(&serde_json::to_vec(&fixture["annotations"]).unwrap())
                .unwrap();
        let engine_frames =
            parse_engine_frames(&serde_json::to_vec(&fixture["engine_frames"]).unwrap()).unwrap();
        let report = evaluate_temporal_localization(
            &annotations,
            &engine_frames,
            EvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(report, fixture["expected"]);
    }

    #[cfg(unix)]
    #[test]
    fn file_evaluation_binds_hash_geometry_and_scene_report() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let video_path = directory.path().join("recording.mp4");
        let annotation_path = directory.path().join("annotations.json");
        let scenes_path = directory.path().join("scenes.jsonl");
        let ffprobe_path = directory.path().join("ffprobe-test-double");
        std::fs::write(&video_path, b"test video bytes").unwrap();
        let (video_hash, _) = hash_file(&video_path, DEFAULT_MAXIMUM_VIDEO_BYTES).unwrap();

        let mut annotation = annotations(vec![annotation_frame(0, Vec::new())]);
        annotation.sequence.video_sha256 = video_hash.clone();
        annotation.sequence.width = 3840;
        annotation.sequence.height = 2160;
        annotation.annotation_metadata = Some(AnnotationMetadata {
            source_annotation_format: "cvat_video_1.1".to_owned(),
            source_annotation_sha256: "b".repeat(64),
            reviewed_frames_sha256: "c".repeat(64),
            epoch_events_sha256: None,
            frame_index_offset: 0,
            source_video_stream_index: 1,
            presentation_time_source: "ffprobe_best_effort_timestamp_time".to_owned(),
            presentation_time_origin_seconds: 0.0,
            ball_radius_method: "mean_box_half_extent".to_owned(),
            cue_polyline_point_order: "first_tip_second_butt".to_owned(),
            tool_version: 1,
        });
        std::fs::write(&annotation_path, serde_json::to_vec(&annotation).unwrap()).unwrap();
        std::fs::write(
            &scenes_path,
            format!(
                "{}\n",
                serde_json::to_string(
                    &serde_json::from_str::<Value>(include_str!(
                        "../contracts/v1/scene-rust-tracked-success.json"
                    ))
                    .unwrap()
                )
                .unwrap()
            ),
        )
        .unwrap();
        std::fs::write(
            &ffprobe_path,
            "#!/bin/sh\ntest \"$4\" = 'v:1' || exit 2\nprintf '%s\\n' '{\"streams\":[{\"width\":3840,\"height\":2160}]}'\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&ffprobe_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&ffprobe_path, permissions).unwrap();

        let report = evaluate_files(
            &video_path,
            &annotation_path,
            &scenes_path,
            &ffprobe_path,
            FileEvaluationConfig::default(),
        )
        .unwrap();

        let annotation_hash = hash_bytes(&std::fs::read(&annotation_path).unwrap());
        let scene_hash = hash_bytes(&std::fs::read(&scenes_path).unwrap());
        assert_eq!(report["metadata"]["video_sha256"], video_hash);
        assert_eq!(report["metadata"]["annotations_sha256"], annotation_hash);
        assert_eq!(report["metadata"]["scene_snapshots_sha256"], scene_hash);
        assert_eq!(report["metadata"]["video_geometry"]["width"], 3840);
        assert_eq!(report["metadata"]["source_video_stream_index"], 1);
        assert_eq!(report["metadata"]["scored_engine_frames"], 1);
        assert_eq!(
            report["metadata"]["rust_monotonic_timestamps_used_for_latency"],
            false
        );
        assert_eq!(
            report["temporal_localization"]["ball_localization"]["false_positives"],
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn ffprobe_output_and_runtime_are_bounded() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let output_script = directory.path().join("ffprobe-large-output");
        std::fs::write(&output_script, "#!/bin/sh\nexec head -c 65537 /dev/zero\n").unwrap();
        let mut permissions = std::fs::metadata(&output_script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&output_script, permissions).unwrap();
        let error = video_geometry(
            &output_script,
            Path::new("unused-video"),
            0,
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(error.to_string().contains("bounded output size"));

        let timeout_script = directory.path().join("ffprobe-sleeps");
        std::fs::write(&timeout_script, "#!/bin/sh\nexec sleep 5\n").unwrap();
        let mut permissions = std::fs::metadata(&timeout_script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&timeout_script, permissions).unwrap();
        let error = video_geometry(
            &timeout_script,
            Path::new("unused-video"),
            0,
            Duration::from_millis(20),
        )
        .unwrap_err();
        assert!(error.to_string().contains("timeout"));
    }

    #[test]
    fn bounded_scene_reader_rejects_oversized_lines_and_invalid_frame_ids() {
        let annotations = annotations(vec![annotation_frame(0, Vec::new())]);
        let mut digest = Sha256::new();
        let error =
            normalize_rust_scene_reader(&b"12345\n"[..], &annotations, 0, 100, 4, &mut digest)
                .unwrap_err();
        assert!(error.to_string().contains("line exceeds"));
        assert!(replay_frame_index("frame-zero").is_err());
        assert!(replay_frame_index("frame-0").is_err());
        assert_eq!(replay_frame_index("frame-0002").unwrap(), 2);
    }

    #[test]
    fn rust_scene_adapter_uses_annotation_pts_and_only_fresh_raw_observations() {
        let mut annotations = annotations(vec![annotation_frame(
            0,
            vec![visible_ball("ball-1", 1882.4, 1024.7, 10.0)],
        )]);
        annotations.sequence.width = 3840;
        annotations.sequence.height = 2160;
        annotations.frames[0].presentation_time_seconds = 0.25;
        annotations.frames[0].expected_source_session_epoch = Some("replay-session-01".to_owned());
        annotations.frames[0].cue = AnnotationCue {
            status: AnnotationCueStatus::Visible,
            tip: Some(Point2d { x: 600.0, y: 400.0 }),
            butt: Some(Point2d { x: 400.0, y: 400.0 }),
        };
        let mut snapshot: Value = serde_json::from_str(include_str!(
            "../contracts/v1/scene-rust-tracked-success.json"
        ))
        .unwrap();
        snapshot["source"]["timestamp"]["ticks"] = json!("41607041");
        snapshot["published_at"]["ticks"] = json!("94901583");
        let line = serde_json::to_vec(&snapshot).unwrap();
        let mut digest = Sha256::new();
        let frames = normalize_rust_scene_reader(
            line.as_slice(),
            &annotations,
            0,
            1024 * 1024,
            64 * 1024,
            &mut digest,
        )
        .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].presentation_time_seconds, 0.25);
        assert_eq!(frames[0].balls.len(), 1);
        assert_eq!(frames[0].balls[0].center.x, 1882.4);
        assert_eq!(frames[0].cue_present, Some(true));
        assert_eq!(frames[0].cue_center.unwrap().x, 500.0);

        snapshot["balls"][0]["status"] = json!("tracked");
        let cached_line = serde_json::to_vec(&snapshot).unwrap();
        let mut digest = Sha256::new();
        let cached = normalize_rust_scene_reader(
            cached_line.as_slice(),
            &annotations,
            0,
            1024 * 1024,
            64 * 1024,
            &mut digest,
        )
        .unwrap();
        assert!(cached[0].balls.is_empty());
    }

    #[test]
    fn unavailable_metrics_remain_explicit() {
        let report = unavailable_temporal_metrics("labels missing");
        assert_eq!(report["status"], "unavailable");
        assert_eq!(report["metrics"], Value::Null);
    }
}
