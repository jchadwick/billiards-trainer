//! Typed Rust representation and semantic validation for scene snapshot v1.
//!
//! The JSON Schema is the language-neutral structural contract. This module
//! adds cross-field invariants that JSON Schema cannot express, such as checking
//! image coordinates against the dimensions carried by that measurement.

pub mod calibration;
pub mod calibration_control;
pub mod calibration_store;
#[cfg(feature = "camera-gstreamer-discovery")]
pub mod camera_discovery;
pub mod capture;
pub mod capture_helper_protocol;
pub mod capture_pipeline;
pub mod capture_runtime;
pub mod capture_session_ack;
pub mod capture_settings_wire;
pub mod capture_transport_benchmark;
pub mod configuration_authority;
mod configuration_http;
pub mod detection_batch;
pub mod detection_batch_receiver;
pub mod frame_preview;
pub mod framing;
pub mod game_session;
mod game_session_http;
pub mod geometry;
#[cfg(feature = "camera-gstreamer-probe")]
pub mod gstreamer_capture_source;
pub mod inference;
pub mod live_application;
pub mod live_audit;
pub mod live_config;
pub mod live_handshake;
pub mod live_supervisor;
pub mod live_timing_accumulator;
pub mod performance_audit_receiver;
pub mod process_resource_sampler;
pub mod public_state;
pub mod public_state_lifecycle;
mod scene_transport;
pub mod server;
pub mod server_audit_summary;
pub mod tracking;
pub mod web_static;

pub use scene_transport::{SceneTransportConfig, SceneTransportStats};

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

pub const SCENE_SNAPSHOT_SCHEMA_VERSION: u8 = 1;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_IDENTIFIER_CHARACTERS: usize = 128;

/// A validated complete observation or annotation snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneSnapshotV1 {
    pub schema_version: u8,
    pub engine_instance_id: String,
    pub source: SourceFrame,
    pub snapshot_revision: u64,
    pub transport_sequence: u64,
    pub update_kind: UpdateKind,
    pub published_at: Timestamp,
    pub revisions: Revisions,
    pub outcome: Outcome,
    #[serde(deserialize_with = "required_option")]
    pub reason: Option<String>,
    pub cue: CueState,
    pub balls: Vec<BallState>,
}

/// Source frame identity is scoped to a camera/source session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFrame {
    pub session_id: String,
    pub frame_id: String,
    pub timestamp: Timestamp,
    pub width_px: u32,
    pub height_px: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Timestamp {
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
    InferenceComplete,
    SnapshotPublication,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Revisions {
    pub configuration: u64,
    pub rectification: u64,
    #[serde(deserialize_with = "required_option")]
    pub calibration: Option<u64>,
    pub calibration_state: CalibrationState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationState {
    Valid,
    Invalid,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateKind {
    Observation,
    Annotation,
    Health,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Success,
    Degraded,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CueState {
    pub status: CueStatus,
    #[serde(deserialize_with = "required_option")]
    pub last_observation: Option<LastObservation>,
    #[serde(deserialize_with = "required_option")]
    pub image_center_px: Option<Point2d>,
    #[serde(deserialize_with = "required_option")]
    pub image_space: Option<ImageSpace>,
    #[serde(deserialize_with = "required_option")]
    pub image_width_px: Option<u32>,
    #[serde(deserialize_with = "required_option")]
    pub image_height_px: Option<u32>,
    #[serde(deserialize_with = "required_option")]
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CueStatus {
    Observed,
    NotObserved,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LastObservation {
    pub session_id: String,
    pub frame_id: String,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSpace {
    Raw,
    Rectified,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallState {
    pub track_id: String,
    pub status: BallStatus,
    pub last_observation: LastObservation,
    #[serde(deserialize_with = "required_option")]
    pub measurement: Option<Measurement>,
    #[serde(deserialize_with = "required_option")]
    pub estimate: Option<Estimate>,
    pub identity: BallIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BallStatus {
    Observed,
    Tracked,
    Occluded,
    Lost,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub image_center_px: Point2d,
    pub image_space: ImageSpace,
    pub image_width_px: u32,
    pub image_height_px: u32,
    #[serde(deserialize_with = "required_option")]
    pub table_center_mm: Option<Point2d>,
    pub detector_confidence: f64,
    pub table_position_uncertainty: PositionUncertainty,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Estimate {
    pub kind: EstimateKind,
    pub as_of: Timestamp,
    #[serde(deserialize_with = "required_option")]
    pub image_center_px: Option<Point2d>,
    #[serde(deserialize_with = "required_option")]
    pub image_space: Option<ImageSpace>,
    #[serde(deserialize_with = "required_option")]
    pub image_width_px: Option<u32>,
    #[serde(deserialize_with = "required_option")]
    pub image_height_px: Option<u32>,
    #[serde(deserialize_with = "required_option")]
    pub table_center_mm: Option<Point2d>,
    pub table_position_uncertainty: PositionUncertainty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateKind {
    Filtered,
    Extrapolated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionUncertainty {
    pub status: UncertaintyStatus,
    #[serde(deserialize_with = "required_option")]
    pub sigma_x_mm: Option<f64>,
    #[serde(deserialize_with = "required_option")]
    pub sigma_y_mm: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyStatus {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallIdentity {
    #[serde(deserialize_with = "required_option")]
    pub number: Option<u8>,
    #[serde(deserialize_with = "required_option")]
    pub confidence: Option<f64>,
    pub evidence: Vec<IdentityEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityEvidence {
    pub source: String,
    #[serde(deserialize_with = "required_option")]
    pub confidence: Option<f64>,
}

/// Deserialize an optional value without treating an omitted key as JSON null.
///
/// The v1 schema requires nullable fields to be present. Serde otherwise treats
/// a missing `Option<T>` field as `None`, which would silently accept an older
/// or incomplete message with different semantics.
fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractValidationError {
    pub path: String,
    pub message: String,
}

impl ContractValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl Display for ContractValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.path, self.message)
    }
}

impl Error for ContractValidationError {}

impl SceneSnapshotV1 {
    /// Validate cross-field and numeric semantics not expressible by the schema.
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        if self.schema_version != SCENE_SNAPSHOT_SCHEMA_VERSION {
            return Err(ContractValidationError::new(
                "schema_version",
                format!("expected {SCENE_SNAPSHOT_SCHEMA_VERSION}"),
            ));
        }
        validate_identifier(&self.engine_instance_id, "engine_instance_id")?;
        validate_identifier(&self.source.session_id, "source.session_id")?;
        validate_identifier(&self.source.frame_id, "source.frame_id")?;
        validate_timestamp(&self.source.timestamp, "source.timestamp")?;
        validate_timestamp(&self.published_at, "published_at")?;
        if self.source.width_px == 0 || self.source.height_px == 0 {
            return Err(ContractValidationError::new(
                "source",
                "source dimensions must be positive",
            ));
        }
        validate_safe_counter(self.snapshot_revision, "snapshot_revision")?;
        validate_safe_counter(self.transport_sequence, "transport_sequence")?;
        validate_safe_counter(self.revisions.configuration, "revisions.configuration")?;
        validate_safe_counter(self.revisions.rectification, "revisions.rectification")?;
        if let Some(calibration) = self.revisions.calibration {
            validate_safe_counter(calibration, "revisions.calibration")?;
        }
        match (self.revisions.calibration_state, self.revisions.calibration) {
            (CalibrationState::Valid, None) => {
                return Err(ContractValidationError::new(
                    "revisions.calibration",
                    "valid calibration requires a revision",
                ));
            }
            (CalibrationState::Unavailable, Some(_)) => {
                return Err(ContractValidationError::new(
                    "revisions.calibration",
                    "unavailable calibration cannot have a revision",
                ));
            }
            _ => {}
        }

        match (self.outcome, self.reason.as_deref()) {
            (Outcome::Success, None) => {}
            (Outcome::Success, Some(_)) => {
                return Err(ContractValidationError::new(
                    "reason",
                    "successful snapshots must not carry a failure reason",
                ));
            }
            (_, None) => {
                return Err(ContractValidationError::new(
                    "reason",
                    "degraded, failed, and skipped snapshots require a reason",
                ));
            }
            (_, Some(reason)) if reason.trim().is_empty() => {
                return Err(ContractValidationError::new(
                    "reason",
                    "non-success outcomes require a non-empty reason",
                ));
            }
            (_, Some(reason)) if reason.chars().count() > 512 => {
                return Err(ContractValidationError::new(
                    "reason",
                    "reason must not exceed 512 characters",
                ));
            }
            (_, Some(_)) => {}
        }

        self.validate_cue()?;
        let mut track_ids = HashSet::with_capacity(self.balls.len());
        for (index, ball) in self.balls.iter().enumerate() {
            let path = format!("balls[{index}]");
            validate_identifier(&ball.track_id, &format!("{path}.track_id"))?;
            if !track_ids.insert(&ball.track_id) {
                return Err(ContractValidationError::new(
                    format!("{path}.track_id"),
                    "track IDs must be unique within one snapshot",
                ));
            }
            validate_observation(&ball.last_observation, &format!("{path}.last_observation"))?;
            if ball.status == BallStatus::Observed && ball.measurement.is_none() {
                return Err(ContractValidationError::new(
                    format!("{path}.measurement"),
                    "observed balls require an image measurement",
                ));
            }
            if ball.status == BallStatus::Observed
                && ball.last_observation.session_id != self.source.session_id
            {
                return Err(ContractValidationError::new(
                    format!("{path}.last_observation.session_id"),
                    "observed ball must belong to the snapshot source session",
                ));
            }
            if ball.status == BallStatus::Observed
                && ball.last_observation.frame_id != self.source.frame_id
            {
                return Err(ContractValidationError::new(
                    format!("{path}.last_observation.frame_id"),
                    "observed ball must match the snapshot source frame",
                ));
            }
            if let Some(measurement) = &ball.measurement {
                validate_measurement(
                    measurement,
                    &format!("{path}.measurement"),
                    &self.source,
                    self.revisions.calibration_state,
                )?;
            }
            if let Some(estimate) = &ball.estimate {
                validate_estimate(
                    estimate,
                    &format!("{path}.estimate"),
                    &self.source,
                    self.revisions.calibration_state,
                )?;
            }
            validate_identity(&ball.identity, &format!("{path}.identity"))?;
        }
        Ok(())
    }

    fn validate_cue(&self) -> Result<(), ContractValidationError> {
        let cue = &self.cue;
        match cue.status {
            CueStatus::Observed => {
                let observation = cue.last_observation.as_ref().ok_or_else(|| {
                    ContractValidationError::new("cue.last_observation", "required when observed")
                })?;
                validate_observation(observation, "cue.last_observation")?;
                let center = cue.image_center_px.as_ref().ok_or_else(|| {
                    ContractValidationError::new("cue.image_center_px", "required when observed")
                })?;
                let image_space = cue.image_space.ok_or_else(|| {
                    ContractValidationError::new("cue.image_space", "required when observed")
                })?;
                let width = cue.image_width_px.ok_or_else(|| {
                    ContractValidationError::new("cue.image_width_px", "required when observed")
                })?;
                let height = cue.image_height_px.ok_or_else(|| {
                    ContractValidationError::new("cue.image_height_px", "required when observed")
                })?;
                validate_frame_dimensions(image_space, width, height, &self.source, "cue")?;
                validate_image_point(center, width, height, "cue.image_center_px")?;
                let confidence = cue.confidence.ok_or_else(|| {
                    ContractValidationError::new("cue.confidence", "required when observed")
                })?;
                validate_probability(confidence, "cue.confidence")?;
            }
            CueStatus::NotObserved => {
                let observation = cue.last_observation.as_ref().ok_or_else(|| {
                    ContractValidationError::new(
                        "cue.last_observation",
                        "negative evidence must identify its source observation",
                    )
                })?;
                validate_observation(observation, "cue.last_observation")?;
                if cue.image_center_px.is_some() || cue.confidence.is_some() {
                    return Err(ContractValidationError::new(
                        "cue",
                        "not_observed cue data must not contain a location or confidence",
                    ));
                }
                let image_space = cue.image_space.ok_or_else(|| {
                    ContractValidationError::new(
                        "cue.image_space",
                        "required for a successful negative observation",
                    )
                })?;
                let (width, height) =
                    validate_image_dimensions(cue.image_width_px, cue.image_height_px, "cue")?;
                validate_frame_dimensions(image_space, width, height, &self.source, "cue")?;
            }
            CueStatus::Unavailable => {
                if cue.last_observation.is_some()
                    || cue.image_center_px.is_some()
                    || cue.image_space.is_some()
                    || cue.image_width_px.is_some()
                    || cue.image_height_px.is_some()
                    || cue.confidence.is_some()
                {
                    return Err(ContractValidationError::new(
                        "cue",
                        "unavailable cue evidence must not carry measurements",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Fail-closed error returned by the public v1 JSON boundary functions.
#[derive(Debug)]
pub enum SceneSnapshotError {
    Json(serde_json::Error),
    Contract(ContractValidationError),
}

impl Display for SceneSnapshotError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid scene snapshot JSON: {error}"),
            Self::Contract(error) => write!(formatter, "invalid scene snapshot: {error}"),
        }
    }
}

impl Error for SceneSnapshotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Contract(error) => Some(error),
        }
    }
}

impl From<serde_json::Error> for SceneSnapshotError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<ContractValidationError> for SceneSnapshotError {
    fn from(error: ContractValidationError) -> Self {
        Self::Contract(error)
    }
}

/// Decode a v1 JSON snapshot and apply all contract invariants.
pub fn decode_scene_snapshot_v1(input: &[u8]) -> Result<SceneSnapshotV1, SceneSnapshotError> {
    let snapshot: SceneSnapshotV1 = serde_json::from_slice(input)?;
    snapshot.validate()?;
    Ok(snapshot)
}

/// Encode a validated v1 snapshot as compact JSON bytes.
pub fn encode_scene_snapshot_v1(snapshot: &SceneSnapshotV1) -> Result<Vec<u8>, SceneSnapshotError> {
    snapshot.validate()?;
    Ok(serde_json::to_vec(snapshot)?)
}

fn validate_identifier(value: &str, path: &str) -> Result<(), ContractValidationError> {
    if value.is_empty()
        || value.chars().count() > MAX_IDENTIFIER_CHARACTERS
        || value.chars().any(char::is_control)
    {
        return Err(ContractValidationError::new(
            path,
            "must be a non-empty identifier without control characters",
        ));
    }
    Ok(())
}

fn validate_safe_counter(value: u64, path: &str) -> Result<(), ContractValidationError> {
    if value > MAX_SAFE_JSON_INTEGER {
        return Err(ContractValidationError::new(
            path,
            "exceeds the exact-integer range of common JSON clients",
        ));
    }
    Ok(())
}

fn validate_timestamp(value: &Timestamp, path: &str) -> Result<(), ContractValidationError> {
    if value.ticks.is_empty()
        || value.ticks.len() > 20
        || !value.ticks.bytes().all(|byte| byte.is_ascii_digit())
        || (value.ticks.len() > 1 && value.ticks.starts_with('0'))
    {
        return Err(ContractValidationError::new(
            format!("{path}.ticks"),
            "must be an unsigned decimal nanosecond string without leading zeros",
        ));
    }
    validate_identifier(&value.domain_id, &format!("{path}.domain_id"))
}

fn validate_observation(
    observation: &LastObservation,
    path: &str,
) -> Result<(), ContractValidationError> {
    validate_identifier(&observation.session_id, &format!("{path}.session_id"))?;
    validate_identifier(&observation.frame_id, &format!("{path}.frame_id"))?;
    validate_timestamp(&observation.timestamp, &format!("{path}.timestamp"))
}

fn validate_probability(value: f64, path: &str) -> Result<(), ContractValidationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ContractValidationError::new(
            path,
            "must be finite and between zero and one",
        ));
    }
    Ok(())
}

fn validate_point(value: &Point2d, path: &str) -> Result<(), ContractValidationError> {
    if !value.x.is_finite() || !value.y.is_finite() {
        return Err(ContractValidationError::new(
            path,
            "coordinates must be finite numbers",
        ));
    }
    Ok(())
}

fn validate_image_dimensions(
    width: Option<u32>,
    height: Option<u32>,
    path: &str,
) -> Result<(u32, u32), ContractValidationError> {
    match (width, height) {
        (Some(width), Some(height)) if width > 0 && height > 0 => Ok((width, height)),
        _ => Err(ContractValidationError::new(
            path,
            "image dimensions must both be positive",
        )),
    }
}

fn validate_image_point(
    point: &Point2d,
    width: u32,
    height: u32,
    path: &str,
) -> Result<(), ContractValidationError> {
    validate_point(point, path)?;
    if point.x < 0.0 || point.y < 0.0 || point.x >= f64::from(width) || point.y >= f64::from(height)
    {
        return Err(ContractValidationError::new(
            path,
            "center must fall inside the declared image dimensions",
        ));
    }
    Ok(())
}

fn validate_frame_dimensions(
    image_space: ImageSpace,
    width: u32,
    height: u32,
    source: &SourceFrame,
    path: &str,
) -> Result<(), ContractValidationError> {
    if width == 0 || height == 0 {
        return Err(ContractValidationError::new(
            path,
            "image dimensions must be positive",
        ));
    }
    if image_space == ImageSpace::Raw && (width != source.width_px || height != source.height_px) {
        return Err(ContractValidationError::new(
            path,
            "raw measurement dimensions must match the source frame",
        ));
    }
    Ok(())
}

fn validate_uncertainty(
    value: &PositionUncertainty,
    path: &str,
) -> Result<(), ContractValidationError> {
    match (value.status, value.sigma_x_mm, value.sigma_y_mm) {
        (UncertaintyStatus::Available, Some(x), Some(y))
            if x.is_finite() && y.is_finite() && x >= 0.0 && y >= 0.0 => Ok(()),
        (UncertaintyStatus::Unavailable, None, None) => Ok(()),
        _ => Err(ContractValidationError::new(
            path,
            "available uncertainty needs finite nonnegative sigmas; unavailable uncertainty needs null sigmas",
        )),
    }
}

fn validate_measurement(
    measurement: &Measurement,
    path: &str,
    source: &SourceFrame,
    calibration_state: CalibrationState,
) -> Result<(), ContractValidationError> {
    validate_frame_dimensions(
        measurement.image_space,
        measurement.image_width_px,
        measurement.image_height_px,
        source,
        path,
    )?;
    validate_image_point(
        &measurement.image_center_px,
        measurement.image_width_px,
        measurement.image_height_px,
        &format!("{path}.image_center_px"),
    )?;
    validate_probability(
        measurement.detector_confidence,
        &format!("{path}.detector_confidence"),
    )?;
    validate_uncertainty(
        &measurement.table_position_uncertainty,
        &format!("{path}.table_position_uncertainty"),
    )?;
    if let Some(center) = &measurement.table_center_mm {
        validate_point(center, &format!("{path}.table_center_mm"))?;
        if calibration_state != CalibrationState::Valid {
            return Err(ContractValidationError::new(
                format!("{path}.table_center_mm"),
                "table coordinates require valid calibration",
            ));
        }
    } else if measurement.table_position_uncertainty.status == UncertaintyStatus::Available {
        return Err(ContractValidationError::new(
            format!("{path}.table_position_uncertainty"),
            "table uncertainty requires a table position",
        ));
    }
    Ok(())
}

fn validate_estimate(
    estimate: &Estimate,
    path: &str,
    source: &SourceFrame,
    calibration_state: CalibrationState,
) -> Result<(), ContractValidationError> {
    validate_timestamp(&estimate.as_of, &format!("{path}.as_of"))?;
    if let Some(center) = &estimate.image_center_px {
        let image_space = estimate.image_space.ok_or_else(|| {
            ContractValidationError::new(
                format!("{path}.image_space"),
                "required when an image estimate is available",
            )
        })?;
        let (width, height) =
            validate_image_dimensions(estimate.image_width_px, estimate.image_height_px, path)?;
        validate_frame_dimensions(image_space, width, height, source, path)?;
        validate_image_point(center, width, height, &format!("{path}.image_center_px"))?;
    } else if estimate.image_space.is_some()
        || estimate.image_width_px.is_some()
        || estimate.image_height_px.is_some()
    {
        return Err(ContractValidationError::new(
            path,
            "image space and dimensions must be unavailable with a null image estimate",
        ));
    }
    validate_uncertainty(
        &estimate.table_position_uncertainty,
        &format!("{path}.table_position_uncertainty"),
    )?;
    if let Some(center) = &estimate.table_center_mm {
        validate_point(center, &format!("{path}.table_center_mm"))?;
        if calibration_state != CalibrationState::Valid {
            return Err(ContractValidationError::new(
                format!("{path}.table_center_mm"),
                "table coordinates require valid calibration",
            ));
        }
    } else if estimate.table_position_uncertainty.status == UncertaintyStatus::Available {
        return Err(ContractValidationError::new(
            format!("{path}.table_position_uncertainty"),
            "table uncertainty requires a table position",
        ));
    }
    Ok(())
}

fn validate_identity(value: &BallIdentity, path: &str) -> Result<(), ContractValidationError> {
    match (value.number, value.confidence) {
        (Some(number), Some(confidence)) if (1..=15).contains(&number) => {
            validate_probability(confidence, &format!("{path}.confidence"))?;
            if value.evidence.is_empty() {
                return Err(ContractValidationError::new(
                    format!("{path}.evidence"),
                    "a numbered identity requires at least one evidence source",
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(ContractValidationError::new(
                path,
                "number and confidence must both be present or both be unavailable",
            ));
        }
    }
    for (index, evidence) in value.evidence.iter().enumerate() {
        let evidence_path = format!("{path}.evidence[{index}]");
        validate_identifier(&evidence.source, &format!("{evidence_path}.source"))?;
        if let Some(confidence) = evidence.confidence {
            validate_probability(confidence, &format!("{evidence_path}.confidence"))?;
        }
    }
    Ok(())
}
