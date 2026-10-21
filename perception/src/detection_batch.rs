//! Typed, fail-closed representation of the frame-scoped detection batch v1.
//!
//! A batch is the complete output for one source frame. Its detection indices
//! identify array entries only within that frame; they are never track IDs.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

pub const DETECTION_BATCH_SCHEMA_VERSION: u8 = 1;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
/// Maximum Unicode scalar count for engine, session, frame, and clock IDs.
pub const MAX_IDENTIFIER_CHARACTERS: usize = 128;
const CENTER_TOLERANCE_PX: f64 = 0.0001;

/// Complete detector result for one source frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectionBatchV1 {
    pub schema_version: u8,
    pub engine_instance_id: String,
    pub batch_sequence: u64,
    pub source: SourceFrame,
    pub image_space: ImageSpace,
    pub model: ModelIdentity,
    pub settings: InferenceSettings,
    pub inference_started_at: BatchTimestamp,
    pub inference_completed_at: BatchTimestamp,
    pub published_at: BatchTimestamp,
    pub outcome: BatchOutcome,
    #[serde(deserialize_with = "required_option")]
    pub reason: Option<String>,
    pub detections: Vec<Detection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFrame {
    pub session_id: String,
    pub frame_id: String,
    /// Null means no trustworthy source-clock timestamp was available.
    #[serde(deserialize_with = "required_option")]
    pub timestamp: Option<BatchTimestamp>,
    pub width_px: u32,
    pub height_px: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchTimestamp {
    /// Decimal-string nanoseconds preserve exact values in JavaScript clients.
    pub ticks: String,
    pub unit: TimeUnit,
    pub clock: ClockKind,
    pub domain_id: String,
    pub stage: TimestampStage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeUnit {
    Nanoseconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockKind {
    Monotonic,
    UnixEpoch,
    Sensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimestampStage {
    Exposure,
    CameraReadComplete,
    SharedMemoryPublication,
    FrameArrival,
    InferenceStarted,
    InferenceComplete,
    BatchPublication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSpace {
    Raw,
    Rectified,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub artifact_sha256: String,
    /// Null when the exported artifact has no verified checkpoint lineage.
    #[serde(deserialize_with = "required_option")]
    pub source_checkpoint_sha256: Option<String>,
    pub classes: Vec<ModelClass>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelClass {
    pub class_id: u32,
    pub class_name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceSettings {
    pub input_width_px: u32,
    pub input_height_px: u32,
    pub confidence_threshold: f64,
    pub iou_threshold: f64,
    pub max_detections: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchOutcome {
    Success,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Detection {
    pub detection_index: u64,
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f64,
    pub bbox_image_px: BoundingBox,
    pub center_image_px: Point,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundingBox {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Cross-field validation error with a JSON-like field path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchValidationError {
    pub path: String,
    pub message: String,
}

impl BatchValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl Display for BatchValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.path, self.message)
    }
}

impl Error for BatchValidationError {}

/// Decode a JSON batch and enforce its structural and semantic invariants.
pub fn decode_detection_batch_v1(input: &[u8]) -> Result<DetectionBatchV1, DetectionBatchError> {
    let batch: DetectionBatchV1 = serde_json::from_slice(input)?;
    batch.validate()?;
    Ok(batch)
}

/// Validate and encode a batch as compact JSON.
pub fn encode_detection_batch_v1(batch: &DetectionBatchV1) -> Result<Vec<u8>, DetectionBatchError> {
    batch.validate()?;
    Ok(serde_json::to_vec(batch)?)
}

impl DetectionBatchV1 {
    /// Check semantic rules that cannot be expressed in JSON Schema alone.
    pub fn validate(&self) -> Result<(), BatchValidationError> {
        if self.schema_version != DETECTION_BATCH_SCHEMA_VERSION {
            return Err(BatchValidationError::new(
                "schema_version",
                format!("expected {DETECTION_BATCH_SCHEMA_VERSION}"),
            ));
        }
        validate_identifier(&self.engine_instance_id, "engine_instance_id")?;
        validate_safe_counter(self.batch_sequence, "batch_sequence")?;
        validate_identifier(&self.source.session_id, "source.session_id")?;
        validate_identifier(&self.source.frame_id, "source.frame_id")?;
        if self.source.width_px == 0 || self.source.height_px == 0 {
            return Err(BatchValidationError::new(
                "source",
                "source dimensions must be positive",
            ));
        }
        if let Some(timestamp) = &self.source.timestamp {
            validate_timestamp(timestamp, "source.timestamp")?;
            if matches!(
                timestamp.stage,
                TimestampStage::InferenceStarted
                    | TimestampStage::InferenceComplete
                    | TimestampStage::BatchPublication
            ) {
                return Err(BatchValidationError::new(
                    "source.timestamp.stage",
                    "source time must identify capture, publication, or frame arrival",
                ));
            }
        }

        validate_model(&self.model)?;
        validate_settings(&self.settings)?;
        validate_inference_timestamp(&self.inference_started_at, "inference_started_at")?;
        validate_inference_timestamp(&self.inference_completed_at, "inference_completed_at")?;
        validate_publication_timestamp(&self.published_at, "published_at")?;
        for (path, timestamp) in [
            ("inference_completed_at", &self.inference_completed_at),
            ("published_at", &self.published_at),
        ] {
            if timestamp.domain_id != self.inference_started_at.domain_id {
                return Err(BatchValidationError::new(
                    path,
                    "must use the inference start clock domain",
                ));
            }
        }
        let started = parse_ticks(&self.inference_started_at, "inference_started_at")?;
        let completed = parse_ticks(&self.inference_completed_at, "inference_completed_at")?;
        let published = parse_ticks(&self.published_at, "published_at")?;
        if completed < started {
            return Err(BatchValidationError::new(
                "inference_completed_at.ticks",
                "must not precede inference start",
            ));
        }
        if published < completed {
            return Err(BatchValidationError::new(
                "published_at.ticks",
                "must not precede inference completion",
            ));
        }

        match (self.outcome, self.reason.as_deref()) {
            (BatchOutcome::Success, None) => {}
            (BatchOutcome::Success, Some(_)) => {
                return Err(BatchValidationError::new(
                    "reason",
                    "successful batches must not have a reason",
                ));
            }
            (_, None) => {
                return Err(BatchValidationError::new(
                    "reason",
                    "failed and skipped batches require a reason",
                ));
            }
            (_, Some(reason)) if reason.trim().is_empty() => {
                return Err(BatchValidationError::new(
                    "reason",
                    "reason must not be empty",
                ));
            }
            (_, Some(reason)) if reason.chars().count() > 512 => {
                return Err(BatchValidationError::new(
                    "reason",
                    "reason must not exceed 512 characters",
                ));
            }
            (_, Some(_)) => {}
        }
        if self.outcome != BatchOutcome::Success && !self.detections.is_empty() {
            return Err(BatchValidationError::new(
                "detections",
                "failed and skipped batches must not carry detections",
            ));
        }
        if self.detections.len() > self.settings.max_detections as usize {
            return Err(BatchValidationError::new(
                "detections",
                "batch exceeds configured maximum detections",
            ));
        }

        for (index, detection) in self.detections.iter().enumerate() {
            validate_detection(detection, index, self)?;
        }
        Ok(())
    }
}

/// Fail-closed error returned by the public JSON boundary functions.
#[derive(Debug)]
pub enum DetectionBatchError {
    Json(serde_json::Error),
    Contract(BatchValidationError),
}

impl Display for DetectionBatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid detection batch JSON: {error}"),
            Self::Contract(error) => write!(formatter, "invalid detection batch: {error}"),
        }
    }
}

impl Error for DetectionBatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Contract(error) => Some(error),
        }
    }
}

impl From<serde_json::Error> for DetectionBatchError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<BatchValidationError> for DetectionBatchError {
    fn from(error: BatchValidationError) -> Self {
        Self::Contract(error)
    }
}

fn validate_detection(
    detection: &Detection,
    index: usize,
    batch: &DetectionBatchV1,
) -> Result<(), BatchValidationError> {
    let path = format!("detections[{index}]");
    validate_safe_counter(
        detection.detection_index,
        &format!("{path}.detection_index"),
    )?;
    if detection.detection_index != index as u64 {
        return Err(BatchValidationError::new(
            format!("{path}.detection_index"),
            "must equal its zero-based position in the batch",
        ));
    }
    let model_class = batch
        .model
        .classes
        .iter()
        .find(|class| class.class_id == detection.class_id)
        .ok_or_else(|| {
            BatchValidationError::new(
                format!("{path}.class_id"),
                "is absent from the model class map",
            )
        })?;
    if model_class.class_name != detection.class_name {
        return Err(BatchValidationError::new(
            format!("{path}.class_name"),
            "does not match the model class map",
        ));
    }
    validate_probability(detection.confidence, &format!("{path}.confidence"))?;
    let bbox = detection.bbox_image_px;
    if ![bbox.left, bbox.top, bbox.right, bbox.bottom]
        .into_iter()
        .all(f64::is_finite)
        || bbox.left < 0.0
        || bbox.top < 0.0
        || bbox.right <= bbox.left
        || bbox.bottom <= bbox.top
        || bbox.right > batch.source.width_px as f64
        || bbox.bottom > batch.source.height_px as f64
    {
        return Err(BatchValidationError::new(
            format!("{path}.bbox_image_px"),
            "must be finite, non-empty, and inside the declared source frame",
        ));
    }
    let center = detection.center_image_px;
    if !center.x.is_finite() || !center.y.is_finite() {
        return Err(BatchValidationError::new(
            format!("{path}.center_image_px"),
            "coordinates must be finite",
        ));
    }
    let expected_x = (bbox.left + bbox.right) / 2.0;
    let expected_y = (bbox.top + bbox.bottom) / 2.0;
    if (center.x - expected_x).abs() > CENTER_TOLERANCE_PX
        || (center.y - expected_y).abs() > CENTER_TOLERANCE_PX
    {
        return Err(BatchValidationError::new(
            format!("{path}.center_image_px"),
            "must be the center of bbox_image_px",
        ));
    }
    Ok(())
}

fn validate_model(model: &ModelIdentity) -> Result<(), BatchValidationError> {
    validate_sha256(&model.artifact_sha256, "model.artifact_sha256")?;
    if let Some(checkpoint_hash) = &model.source_checkpoint_sha256 {
        validate_sha256(checkpoint_hash, "model.source_checkpoint_sha256")?;
    }
    if model.classes.is_empty() {
        return Err(BatchValidationError::new(
            "model.classes",
            "class map must not be empty",
        ));
    }
    let mut ids = HashSet::with_capacity(model.classes.len());
    let mut names = HashSet::with_capacity(model.classes.len());
    for (index, class) in model.classes.iter().enumerate() {
        let path = format!("model.classes[{index}]");
        validate_class_name(&class.class_name, &format!("{path}.class_name"))?;
        if !ids.insert(class.class_id) {
            return Err(BatchValidationError::new(
                format!("{path}.class_id"),
                "class IDs must be unique",
            ));
        }
        if !names.insert(&class.class_name) {
            return Err(BatchValidationError::new(
                format!("{path}.class_name"),
                "class names must be unique",
            ));
        }
    }
    Ok(())
}

fn validate_settings(settings: &InferenceSettings) -> Result<(), BatchValidationError> {
    if settings.input_width_px == 0 || settings.input_height_px == 0 || settings.max_detections == 0
    {
        return Err(BatchValidationError::new(
            "settings",
            "input dimensions and maximum detections must be positive",
        ));
    }
    validate_probability(
        settings.confidence_threshold,
        "settings.confidence_threshold",
    )?;
    validate_probability(settings.iou_threshold, "settings.iou_threshold")?;
    Ok(())
}

fn validate_inference_timestamp(
    timestamp: &BatchTimestamp,
    path: &str,
) -> Result<(), BatchValidationError> {
    validate_timestamp(timestamp, path)?;
    if timestamp.clock != ClockKind::Monotonic {
        return Err(BatchValidationError::new(
            format!("{path}.clock"),
            "inference timing must use a monotonic clock",
        ));
    }
    if !matches!(
        timestamp.stage,
        TimestampStage::InferenceStarted | TimestampStage::InferenceComplete
    ) {
        return Err(BatchValidationError::new(
            format!("{path}.stage"),
            "must identify an inference stage",
        ));
    }
    Ok(())
}

fn validate_publication_timestamp(
    timestamp: &BatchTimestamp,
    path: &str,
) -> Result<(), BatchValidationError> {
    validate_timestamp(timestamp, path)?;
    if timestamp.clock != ClockKind::Monotonic {
        return Err(BatchValidationError::new(
            format!("{path}.clock"),
            "publication timing must use a monotonic clock",
        ));
    }
    if timestamp.stage != TimestampStage::BatchPublication {
        return Err(BatchValidationError::new(
            format!("{path}.stage"),
            "must identify batch publication",
        ));
    }
    Ok(())
}

fn validate_timestamp(timestamp: &BatchTimestamp, path: &str) -> Result<(), BatchValidationError> {
    validate_identifier(&timestamp.domain_id, &format!("{path}.domain_id"))?;
    parse_ticks(timestamp, path)?;
    Ok(())
}

fn parse_ticks(timestamp: &BatchTimestamp, path: &str) -> Result<u64, BatchValidationError> {
    timestamp.ticks.parse::<u64>().map_err(|_| {
        BatchValidationError::new(
            format!("{path}.ticks"),
            "must be an unsigned 64-bit decimal nanosecond count",
        )
    })
}

fn validate_sha256(value: &str, path: &str) -> Result<(), BatchValidationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(BatchValidationError::new(
            path,
            "must be a lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_class_name(value: &str, path: &str) -> Result<(), BatchValidationError> {
    if value.trim().is_empty()
        || value.chars().count() > MAX_IDENTIFIER_CHARACTERS
        || value.chars().any(char::is_control)
    {
        return Err(BatchValidationError::new(
            path,
            "must be a non-empty class name without control characters",
        ));
    }
    Ok(())
}

fn validate_identifier(value: &str, path: &str) -> Result<(), BatchValidationError> {
    if value.is_empty()
        || value.chars().count() > MAX_IDENTIFIER_CHARACTERS
        || value.chars().any(char::is_control)
    {
        return Err(BatchValidationError::new(
            path,
            "must be a non-empty identifier without control characters",
        ));
    }
    Ok(())
}

fn validate_safe_counter(value: u64, path: &str) -> Result<(), BatchValidationError> {
    if value > MAX_SAFE_JSON_INTEGER {
        return Err(BatchValidationError::new(
            path,
            "must not exceed JavaScript's exact integer range",
        ));
    }
    Ok(())
}

fn validate_probability(value: f64, path: &str) -> Result<(), BatchValidationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(BatchValidationError::new(
            path,
            "must be finite and between zero and one",
        ));
    }
    Ok(())
}

/// Preserve the distinction between a required nullable field and a missing key.
fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}
