//! Compatibility conversion from validated scenes to the existing public state.
//!
//! This boundary intentionally has no transport, clock, cache, or inference
//! dependency. It preserves per-ball observation identity and never promotes a
//! cached position to a detection. A caller that accepts association-capacity
//! degradation must explicitly normalize that *specific* validated outcome;
//! arbitrary degraded observations are never fresh localization here.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{error::Error, fmt};

use crate::{
    BallStatus, ClockKind, ContractValidationError, IdentityEvidence, ImageSpace, Outcome,
    SceneSnapshotV1, Timestamp, TimestampStage, UpdateKind,
};

const RADIUS_SOURCE: &str = "legacy_standard_pool_ball_assumption";

/// Rendering compatibility geometry; radius is an assumption, not a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicStateConfig {
    pub canonical_width_px: u32,
    pub canonical_height_px: u32,
    pub assumed_ball_radius_px: f64,
}

impl Default for PublicStateConfig {
    fn default() -> Self {
        // Versioned compatibility policy is shared with differential fixtures.
        static POLICY: std::sync::OnceLock<PublicStateConfig> = std::sync::OnceLock::new();
        *POLICY.get_or_init(|| {
            let policy: Self = serde_json::from_str(include_str!(
                "../contracts/public-state/legacy-policy-v1.json"
            ))
            .expect("checked-in public-state policy must deserialize");
            policy
                .validate()
                .expect("checked-in public-state policy must validate");
            policy
        })
    }
}

impl PublicStateConfig {
    pub fn validate(&self) -> Result<(), PublicStateError> {
        if self.canonical_width_px == 0
            || self.canonical_height_px == 0
            || !self.assumed_ball_radius_px.is_finite()
            || self.assumed_ball_radius_px <= 0.0
        {
            return Err(PublicStateError::Compatibility(
                "canonical dimensions and assumed radius must be positive and finite".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum PublicStateError {
    Contract(ContractValidationError),
    Compatibility(String),
}
impl fmt::Display for PublicStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(e) => e.fmt(f),
            Self::Compatibility(e) => f.write_str(e),
        }
    }
}
impl Error for PublicStateError {}
impl From<ContractValidationError> for PublicStateError {
    fn from(e: ContractValidationError) -> Self {
        Self::Contract(e)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneStatePayload {
    pub balls: Vec<PublicBall>,
    pub observation_status: Value,
    pub table: Option<Value>,
    pub game_type: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicPosition {
    pub x: f64,
    pub y: f64,
    pub scale: [f64; 2],
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicTablePosition {
    pub x: f64,
    pub y: f64,
    pub unit: String,
    pub origin: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceObservation {
    pub engine_instance_id: String,
    pub session_id: String,
    pub frame_id: String,
    pub timestamp: Timestamp,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicBall {
    pub id: String,
    pub position: PublicPosition,
    #[serde(serialize_with = "serialize_radius")]
    pub radius: f64,
    pub radius_scale: [f64; 2],
    pub radius_unit: String,
    pub radius_source: String,
    pub confidence: f64,
    pub confidence_kind: String,
    pub last_observation_confidence: Option<f64>,
    pub number: Option<u8>,
    #[serde(rename = "type")]
    pub ball_type: String,
    pub identity_status: String,
    pub identity_confidence: Option<f64>,
    pub identity_evidence: Vec<IdentityEvidence>,
    pub is_observed: bool,
    pub tracking_status: BallStatus,
    pub position_kind: String,
    pub source_observation: SourceObservation,
    pub image_space: ImageSpace,
    pub image_width_px: u32,
    pub image_height_px: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table_position: Option<PublicTablePosition>,
}

pub struct SceneSnapshotStateAdapter {
    config: PublicStateConfig,
}
impl SceneSnapshotStateAdapter {
    pub fn new(config: PublicStateConfig) -> Result<Self, PublicStateError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Convert a complete successful localization; non-localization updates have
    /// no fresh payload. Validation also prevents calibrated coordinates with an
    /// invalid/unavailable revision from escaping into the public contract.
    pub fn adapt(
        &self,
        snapshot: &SceneSnapshotV1,
        table: Option<Value>,
        game_type: Option<String>,
    ) -> Result<Option<SceneStatePayload>, PublicStateError> {
        snapshot.validate()?;
        if snapshot.update_kind != UpdateKind::Observation || snapshot.outcome != Outcome::Success {
            return Ok(None);
        }
        let source = &snapshot.source;
        if !matches!(
            source.timestamp.stage,
            TimestampStage::Exposure
                | TimestampStage::CameraReadComplete
                | TimestampStage::SharedMemoryPublication
                | TimestampStage::FrameArrival
        ) {
            return Err(PublicStateError::Compatibility(
                "source timestamp must identify capture or frame publication".into(),
            ));
        }
        let mut balls = Vec::with_capacity(snapshot.balls.len());
        let mut omitted = Vec::new();
        for ball in &snapshot.balls {
            let observed = ball.status == BallStatus::Observed;
            let (point, space, width, height, table_point, kind) = if !observed
                && ball
                    .estimate
                    .as_ref()
                    .is_some_and(|e| e.image_center_px.is_some())
            {
                let e = ball.estimate.as_ref().expect("checked estimate");
                (
                    e.image_center_px.as_ref(),
                    e.image_space,
                    e.image_width_px,
                    e.image_height_px,
                    e.table_center_mm.as_ref(),
                    match e.kind {
                        crate::EstimateKind::Filtered => "filtered",
                        crate::EstimateKind::Extrapolated => "extrapolated",
                    },
                )
            } else if let Some(m) = &ball.measurement {
                (
                    Some(&m.image_center_px),
                    Some(m.image_space),
                    Some(m.image_width_px),
                    Some(m.image_height_px),
                    m.table_center_mm.as_ref(),
                    if observed {
                        "measured"
                    } else {
                        "last_observed"
                    },
                )
            } else {
                omitted.push(ball.track_id.clone());
                continue;
            };
            let (Some(point), Some(ImageSpace::Raw), Some(width), Some(height)) =
                (point, space, width, height)
            else {
                return Err(PublicStateError::Compatibility(
                    "retained preview requires raw coordinates and dimensions".into(),
                ));
            };
            if (width, height) != (source.width_px, source.height_px) {
                return Err(PublicStateError::Compatibility(
                    "image dimensions differ from the source frame".into(),
                ));
            }
            let confidence = ball.measurement.as_ref().map(|m| m.detector_confidence);
            balls.push(PublicBall {
                id: ball.track_id.clone(),
                position: PublicPosition {
                    x: point.x
                        * (f64::from(self.config.canonical_width_px) / f64::from(source.width_px)),
                    y: point.y
                        * (f64::from(self.config.canonical_height_px)
                            / f64::from(source.height_px)),
                    scale: [1.0, 1.0],
                },
                radius: self.config.assumed_ball_radius_px,
                radius_scale: [1.0, 1.0],
                radius_unit: "canonical_pixels".to_owned(),
                radius_source: RADIUS_SOURCE.to_owned(),
                confidence: if observed {
                    confidence.unwrap_or(0.0)
                } else {
                    0.0
                },
                confidence_kind: if observed {
                    "detector_observation".to_owned()
                } else {
                    "unavailable".to_owned()
                },
                last_observation_confidence: confidence,
                number: ball.identity.number,
                ball_type: match ball.identity.number {
                    None => "unknown".to_owned(),
                    Some(8) => "eight".to_owned(),
                    Some(9..=15) => "stripe".to_owned(),
                    _ => "solid".to_owned(),
                },
                identity_status: if ball.identity.number.is_some() {
                    "identified".to_owned()
                } else {
                    "unknown".to_owned()
                },
                identity_confidence: ball.identity.confidence,
                identity_evidence: ball.identity.evidence.clone(),
                is_observed: observed,
                tracking_status: ball.status,
                position_kind: kind.into(),
                source_observation: SourceObservation {
                    engine_instance_id: snapshot.engine_instance_id.clone(),
                    session_id: ball.last_observation.session_id.clone(),
                    frame_id: ball.last_observation.frame_id.clone(),
                    timestamp: ball.last_observation.timestamp.clone(),
                },
                image_space: ImageSpace::Raw,
                image_width_px: width,
                image_height_px: height,
                table_position: table_point.map(|p| PublicTablePosition {
                    x: p.x,
                    y: p.y,
                    unit: "millimeters".to_owned(),
                    origin: "playing_surface_center".to_owned(),
                }),
            });
        }
        let position_status = if !omitted.is_empty() && balls.is_empty() {
            "unavailable"
        } else if !omitted.is_empty() || balls.iter().any(|b| !b.is_observed) {
            "cached"
        } else {
            "observed"
        };
        let timestamp = &source.timestamp;
        let clock = serde_json::to_value(timestamp.clock).expect("enum serialization");
        let stage = serde_json::to_value(timestamp.stage).expect("enum serialization");
        let timestamp_kind = format!(
            "{}:{}:{}",
            clock.as_str().unwrap(),
            timestamp.domain_id,
            stage.as_str().unwrap()
        );
        let (age, age_status) = frame_age(timestamp, &snapshot.published_at);
        let mut status = json!({
            "phase":"rust_scene_observation", "source_session_id":source.session_id, "source_frame_id":source.frame_id,
            "source_timestamp":timestamp.ticks, "source_timestamp_unit":"nanoseconds", "source_timestamp_kind":timestamp_kind,
            "ball_observation_source_timestamp":timestamp.ticks, "ball_observation_source_timestamp_unit":"nanoseconds", "ball_observation_source_timestamp_kind":timestamp_kind,
            "source_timestamp_clock":timestamp.clock, "source_timestamp_domain_id":timestamp.domain_id, "source_timestamp_is_exposure":timestamp.stage == TimestampStage::Exposure,
            "published_at":snapshot.published_at, "frame_publication_age_status":age_status, "frame_age_basis":"source_timestamp_to_snapshot_publication",
            "engine_instance_id":snapshot.engine_instance_id, "snapshot_revision":snapshot.snapshot_revision, "transport_sequence":snapshot.transport_sequence,
            "configuration_revision":snapshot.revisions.configuration, "rectification_revision":snapshot.revisions.rectification, "calibration_revision":snapshot.revisions.calibration, "calibration_state":snapshot.revisions.calibration_state,
            "source_image_space":"raw", "source_width_px":source.width_px, "source_height_px":source.height_px,
            "ball_positions_status":position_status, "ball_detection_status":"observed", "ball_radius_source":RADIUS_SOURCE,
            "omitted_track_ids_without_image_position":omitted, "cue_detection_status":snapshot.cue.status,
            "observation_identity":{ "engine_instance_id":snapshot.engine_instance_id, "session_id":source.session_id, "frame_id":source.frame_id, "snapshot_revision":snapshot.snapshot_revision }
        });
        if let Some(age) = age {
            status["frame_publication_age_seconds"] = json!(age);
        }
        Ok(Some(SceneStatePayload {
            balls,
            observation_status: status,
            table,
            game_type,
        }))
    }
}

fn frame_age(source: &Timestamp, publication: &Timestamp) -> (Option<f64>, &'static str) {
    if source.clock != ClockKind::Monotonic
        || publication.clock != ClockKind::Monotonic
        || publication.stage != TimestampStage::SnapshotPublication
    {
        return (None, "comparable_monotonic_clock_unavailable");
    }
    if source.domain_id != publication.domain_id {
        return (None, "clock_domain_mismatch");
    }
    // Typed contract permits twenty decimal digits (slightly wider than u64).
    let source_ticks: u128 = source.ticks.parse().expect("validated decimal timestamp");
    let publication_ticks: u128 = publication
        .ticks
        .parse()
        .expect("validated decimal timestamp");
    match publication_ticks.checked_sub(source_ticks) {
        Some(ns) => (Some(ns as f64 / 1_000_000_000.0), "available"),
        None => (None, "publication_precedes_source"),
    }
}

// The legacy policy defines an integer radius. Preserve its JSON shape while
// still permitting explicitly configured fractional canonical-pixel radii.
fn serialize_radius<S: serde::Serializer>(radius: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    if radius.fract() == 0.0 && *radius <= 9_007_199_254_740_991.0 {
        serializer.serialize_u64(*radius as u64)
    } else {
        serializer.serialize_f64(*radius)
    }
}
