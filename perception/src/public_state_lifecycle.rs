//! Bounded, latest-only ownership of public scene state.
//!
//! Conversion is pure (`public_state`); this module adds source epochs,
//! monotonic receipt freshness, duplicate/order rejection and stale-cache
//! expiry. A supervisor must explicitly activate the producer identity after
//! process start/restart. Packets from a prior identity are never adopted just
//! because their counters happen to be larger.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::public_state::{
    PublicStateConfig, PublicStateError, SceneSnapshotStateAdapter, SceneStatePayload,
};
use crate::{ClockKind, Outcome, SceneSnapshotV1, Timestamp, UpdateKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateLifecycleConfig {
    pub stale_after_ns: u128,
    pub expire_after_ns: u128,
    pub maximum_balls: usize,
}

impl Default for StateLifecycleConfig {
    fn default() -> Self {
        static POLICY: std::sync::OnceLock<StateLifecycleConfig> = std::sync::OnceLock::new();
        *POLICY.get_or_init(|| {
            let config: Self = serde_json::from_str(include_str!(
                "../contracts/public-state/lifecycle-policy-v1.json"
            ))
            .expect("checked-in state lifecycle policy must deserialize");
            config
                .validate()
                .expect("checked-in state lifecycle policy must validate");
            config
        })
    }
}

impl StateLifecycleConfig {
    pub fn validate(&self) -> Result<(), StateLifecycleError> {
        if self.stale_after_ns == 0
            || self.expire_after_ns <= self.stale_after_ns
            || self.maximum_balls == 0
        {
            return Err(StateLifecycleError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentSceneState {
    /// Public compatibility payload for both HTTP bootstrap and WebSocket state.
    pub payload: SceneStatePayload,
    /// Increments for each explicitly activated producer epoch.
    pub epoch: u64,
    /// Local monotonic receipt time of the last accepted scene event.
    pub updated_monotonic_ns: u128,
    /// Local monotonic receipt time of the last successful localization.
    pub localization_received_monotonic_ns: Option<u128>,
    /// Age of the last successful localization at the requested read time.
    pub localization_age_ns: Option<u128>,
    /// Per-ball receipt ages preserve the age of reused/cached observations.
    pub ball_age_ns: HashMap<String, u128>,
    pub freshness: StateFreshness,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateFreshness {
    Fresh,
    Stale,
    Expired,
}

/// Versioned live delivery envelope. The raw v1 source scene stays intact for
/// training/diagnostic consumers, while `current_state` is computed by Rust's
/// lifecycle and is the gateway's authoritative client-facing state.
#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicStateDeliveryV1<'a> {
    pub delivery_type: &'static str,
    pub delivery_schema_version: u8,
    pub source_snapshot: &'a SceneSnapshotV1,
    pub current_state: Option<&'a CurrentSceneState>,
}

/// Self-contained phase, trust, and independent cue evidence for one live
/// delivery. Ages are durations from the sender and are rebased by the gateway
/// on receipt; no process-local monotonic timestamp crosses this boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerceptionStateV1 {
    pub event_sequence: u64,
    pub image_width_px: Option<u32>,
    pub image_height_px: Option<u32>,
    pub phase: String,
    pub layout_revision: u64,
    pub layout_trusted: bool,
    /// Detector frame whose positively covered full-table scan most recently
    /// established the currently trusted layout. The gateway uses this to
    /// reject a trust-only update when that scene was dropped in transit.
    #[serde(default)]
    pub layout_basis_frame_id: Option<String>,
    /// True only after the phase scheduler has accepted the configured
    /// number of consistent empty full-table observations after quiet dwell.
    /// Raw empty `source_snapshot` values remain available independently.
    pub empty_layout_verified: bool,
    pub guidance_valid: bool,
    pub trust_reason: String,
    pub last_ball_observation_age_ms: Option<u64>,
    pub latest_motion_check_age_ms: Option<u64>,
    pub cue_evidence_age_ms: Option<u64>,
    pub motion_evidence_max_age_ms: u64,
    pub cue_evidence_max_age_ms: u64,
    pub layout_verification_max_age_ms: u64,
    pub cue: Option<CueGeometryStateV1>,
    /// Exact acquisition frame on which this decision was evaluated. Missing
    /// values cannot be joined to spatial video overlays.
    #[serde(default)]
    pub decision_frame_id: Option<String>,
    #[serde(default)]
    pub decision_calibration_revision: Option<u64>,
    #[serde(default)]
    pub decision_rectification_revision: Option<u64>,
    /// Explicit calibrated-table visibility evidence. Absence is treated as
    /// unavailable by newer consumers and never enables guidance.
    #[serde(default)]
    pub table_visibility: Option<TableVisibilityStateV1>,
    /// Exact active image-to-table mapping used for measured production centers.
    /// This is runtime evidence, not a copy of static corner inputs.
    #[serde(default)]
    pub table_projection: Option<TableProjectionV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableProjectionV1 {
    pub direction: String,
    pub image_space: String,
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub camera_profile_id: String,
    pub calibration_revision: u64,
    pub rectification_revision: u64,
    pub table_width_mm: f64,
    pub table_height_mm: f64,
    pub unit: String,
    pub origin: String,
    pub x_axis: String,
    pub y_axis: String,
    pub pixel_to_table_mm: [[f64; 3]; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableVisibilityStateV1 {
    pub status: TableVisibilityStatusV1,
    pub reason: String,
    pub observation_frame_id: String,
    pub observation_age_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TableVisibilityStatusV1 {
    Visible,
    NotVisible,
    Unknown,
}

/// Image-space shaft geometry from the independent cue update loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CueGeometryStateV1 {
    pub shaft_start_px: [f64; 2],
    pub shaft_end_px: [f64; 2],
    pub axis_unit: [f64; 2],
    pub confidence: f64,
    pub observation_age_ms: u64,
    pub signed_direction: Option<String>,
    /// Capture frame from which this independent cue estimate was computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
}

/// Versioned live envelope for a ball observation or a cue/phase-only update.
/// A missing `source_snapshot` never implies a new ball observation.
#[derive(Debug, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PublicStateDeliveryV2<'a> {
    pub delivery_type: &'static str,
    pub delivery_schema_version: u8,
    pub delivery_sequence: u64,
    pub engine_instance_id: &'a str,
    pub source_session_id: &'a str,
    pub source_snapshot: Option<&'a SceneSnapshotV1>,
    pub current_state: Option<&'a CurrentSceneState>,
    pub perception: &'a PerceptionStateV1,
}

impl<'a> PublicStateDeliveryV2<'a> {
    pub fn new(
        delivery_sequence: u64,
        engine_instance_id: &'a str,
        source_session_id: &'a str,
        source_snapshot: Option<&'a SceneSnapshotV1>,
        current_state: Option<&'a CurrentSceneState>,
        perception: &'a PerceptionStateV1,
    ) -> Self {
        Self {
            delivery_type: "billiards_public_state_delivery",
            delivery_schema_version: 2,
            delivery_sequence,
            engine_instance_id,
            source_session_id,
            source_snapshot,
            current_state,
            perception,
        }
    }
}

impl<'a> PublicStateDeliveryV1<'a> {
    pub fn new(
        source_snapshot: &'a SceneSnapshotV1,
        current_state: Option<&'a CurrentSceneState>,
    ) -> Self {
        Self {
            delivery_type: "billiards_public_state_delivery",
            delivery_schema_version: 1,
            source_snapshot,
            current_state,
        }
    }
}
impl StateFreshness {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Stale => "stale",
            Self::Expired => "expired",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestOutcome {
    LocalizationUpdated,
    HealthUpdated,
    AnnotationIgnored,
}

/// Non-scene values carried with a process-boundary delivery. Durations are
/// elapsed ages from the sender and are rebased onto receiver-local monotonic
/// timestamps; no remote clock instant is accepted here.
#[derive(Debug, Clone, Default)]
pub struct SourceStateMetadata {
    pub localization_age_offset_ns: u128,
    pub ball_age_offsets_ns: HashMap<String, u128>,
    pub table: Option<Value>,
    pub game_type: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StateLifecycleStats {
    pub accepted_localizations: u64,
    pub accepted_health_updates: u64,
    pub ignored_annotations: u64,
    pub rejected_identity: u64,
    pub rejected_order: u64,
    pub rejected_duplicate_observation: u64,
    pub rejected_timestamp: u64,
    pub stale_transitions: u64,
    pub expired_transitions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProducerIdentity {
    engine_instance_id: String,
    session_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservationTimestamp {
    clock: ClockKind,
    domain_id: String,
    ticks: u128,
}

#[derive(Debug, Clone)]
struct StoredState {
    payload: SceneStatePayload,
    updated_monotonic_ns: u128,
    localization_received_monotonic_ns: Option<u128>,
    /// Elapsed localization age reported by the process that produced this
    /// state when it crossed a process boundary. It is a duration, not a
    /// comparable monotonic timestamp.
    localization_age_offset_ns: u128,
    ball_received_monotonic_ns: HashMap<String, u128>,
    ball_age_offset_ns: HashMap<String, u128>,
    health_forced_stale: bool,
    health_reason: Option<String>,
    stale_counted: bool,
    expired_counted: bool,
}

/// One source owns one sequence space at a time. `activate_source` is the only
/// operation that can reset its ordering watermarks and it clears old state.
pub struct SceneStateLifecycle {
    config: StateLifecycleConfig,
    adapter: SceneSnapshotStateAdapter,
    epoch: u64,
    active: Option<ProducerIdentity>,
    last_transport_sequence: Option<u64>,
    last_snapshot_revision: Option<u64>,
    last_event_received_monotonic_ns: Option<u128>,
    last_observation: Option<(String, String)>,
    last_observation_timestamp: Option<ObservationTimestamp>,
    latest_perception_status: Option<Value>,
    perception_received_monotonic_ns: Option<u128>,
    stored: Option<StoredState>,
    stats: StateLifecycleStats,
}

impl Default for SceneStateLifecycle {
    fn default() -> Self {
        Self::new(
            StateLifecycleConfig::default(),
            PublicStateConfig::default(),
        )
        .expect("checked-in lifecycle and public-state policies must validate")
    }
}

impl SceneStateLifecycle {
    pub fn new(
        config: StateLifecycleConfig,
        public_state_config: PublicStateConfig,
    ) -> Result<Self, StateLifecycleError> {
        config.validate()?;
        Ok(Self {
            config,
            adapter: SceneSnapshotStateAdapter::new(public_state_config)?,
            epoch: 0,
            active: None,
            last_transport_sequence: None,
            last_snapshot_revision: None,
            last_event_received_monotonic_ns: None,
            last_observation: None,
            last_observation_timestamp: None,
            latest_perception_status: None,
            perception_received_monotonic_ns: None,
            stored: None,
            stats: StateLifecycleStats::default(),
        })
    }

    /// Start a fresh producer epoch after the supervisor has selected the new
    /// engine/session identity. This drops every old position and watermark.
    pub fn activate_source(
        &mut self,
        engine_instance_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<u64, StateLifecycleError> {
        let identity = ProducerIdentity {
            engine_instance_id: engine_instance_id.into(),
            session_id: session_id.into(),
        };
        if identity.engine_instance_id.trim().is_empty()
            || identity.session_id.trim().is_empty()
            || identity.engine_instance_id.len() > 128
            || identity.session_id.len() > 128
        {
            return Err(StateLifecycleError::InvalidIdentity);
        }
        if self.active.as_ref() == Some(&identity) {
            return Err(StateLifecycleError::SourceAlreadyActive);
        }
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or(StateLifecycleError::EpochExhausted)?;
        self.active = Some(identity);
        self.last_transport_sequence = None;
        self.last_snapshot_revision = None;
        self.last_event_received_monotonic_ns = None;
        self.last_observation = None;
        self.last_observation_timestamp = None;
        self.latest_perception_status = None;
        self.perception_received_monotonic_ns = None;
        self.stored = None;
        Ok(self.epoch)
    }

    pub fn stats(&self) -> StateLifecycleStats {
        self.stats
    }

    /// Update phase and cue evidence without renewing any ball measurement or
    /// localization timestamp. Evidence ages continue advancing on reads, and
    /// expired motion/cue/layout evidence fails closed for guidance.
    pub fn update_perception_status(
        &mut self,
        engine_instance_id: &str,
        session_id: &str,
        status: Value,
        received_monotonic_ns: u128,
        now_monotonic_ns: u128,
    ) -> Result<(), StateLifecycleError> {
        let expected = self
            .active
            .as_ref()
            .ok_or(StateLifecycleError::SourceNotActivated)?;
        if expected.engine_instance_id != engine_instance_id || expected.session_id != session_id {
            return Err(StateLifecycleError::UnexpectedIdentity);
        }
        if !status.is_object() {
            return Err(StateLifecycleError::InvalidPerceptionStatus);
        }
        if received_monotonic_ns > now_monotonic_ns
            || self
                .last_event_received_monotonic_ns
                .is_some_and(|previous| received_monotonic_ns < previous)
            || self
                .stored
                .as_ref()
                .is_some_and(|stored| received_monotonic_ns < stored.updated_monotonic_ns)
        {
            return Err(StateLifecycleError::ClockMovedBackwards);
        }
        self.latest_perception_status = Some(status);
        self.perception_received_monotonic_ns = Some(received_monotonic_ns);
        self.last_event_received_monotonic_ns = Some(received_monotonic_ns);
        if let Some(stored) = self.stored.as_mut() {
            stored.updated_monotonic_ns = received_monotonic_ns;
        } else {
            // Cue and phase evidence has its own update cadence and can arrive
            // before the first successful ball localization. Publish an
            // explicit empty, unavailable layout so subscribers can display
            // the independent cue evidence without implying that any balls
            // were measured or that a layout is trustworthy.
            let expected = self
                .active
                .as_ref()
                .expect("source identity was validated above");
            self.stored = Some(StoredState {
                payload: SceneStatePayload {
                    balls: Vec::new(),
                    observation_status: json!({
                        "phase": "rust_perception",
                        "engine_instance_id": expected.engine_instance_id,
                        "source_session_id": expected.session_id,
                        "ball_positions_status": "unavailable",
                        "ball_detection_status": "unavailable",
                        "freshness_status": "stale",
                        "localization_age_ns": Value::Null
                    }),
                    table: None,
                    game_type: None,
                },
                updated_monotonic_ns: received_monotonic_ns,
                localization_received_monotonic_ns: None,
                localization_age_offset_ns: 0,
                ball_received_monotonic_ns: HashMap::new(),
                ball_age_offset_ns: HashMap::new(),
                health_forced_stale: false,
                health_reason: None,
                stale_counted: false,
                expired_counted: false,
            });
        }
        Ok(())
    }

    /// Accept one source event using local monotonic receipt time. Gaps are
    /// allowed because latest-only delivery intentionally drops old frames.
    pub fn ingest(
        &mut self,
        snapshot: &SceneSnapshotV1,
        received_monotonic_ns: u128,
        now_monotonic_ns: u128,
        table: Option<Value>,
        game_type: Option<String>,
    ) -> Result<IngestOutcome, StateLifecycleError> {
        self.ingest_with_source_metadata(
            snapshot,
            received_monotonic_ns,
            now_monotonic_ns,
            SourceStateMetadata {
                table,
                game_type,
                ..SourceStateMetadata::default()
            },
        )
    }

    /// Accept a scene whose source-side age durations are known but whose
    /// source monotonic clock epoch is not comparable to this process. The
    /// elapsed durations are carried as offsets from the receiver's local
    /// receipt timestamps, never as remote clock instants.
    pub fn ingest_with_source_metadata(
        &mut self,
        snapshot: &SceneSnapshotV1,
        received_monotonic_ns: u128,
        now_monotonic_ns: u128,
        source_metadata: SourceStateMetadata,
    ) -> Result<IngestOutcome, StateLifecycleError> {
        let SourceStateMetadata {
            localization_age_offset_ns,
            ball_age_offsets_ns,
            table,
            game_type,
        } = source_metadata;
        snapshot
            .validate()
            .map_err(|error| StateLifecycleError::PublicState(PublicStateError::Contract(error)))?;
        if received_monotonic_ns > now_monotonic_ns {
            return Err(StateLifecycleError::ClockMovedBackwards);
        }
        if self
            .last_event_received_monotonic_ns
            .is_some_and(|previous| received_monotonic_ns < previous)
        {
            return Err(StateLifecycleError::ClockMovedBackwards);
        }
        let expected = self
            .active
            .as_ref()
            .ok_or(StateLifecycleError::SourceNotActivated)?;
        if snapshot.engine_instance_id != expected.engine_instance_id
            || snapshot.source.session_id != expected.session_id
        {
            self.stats.rejected_identity = self.stats.rejected_identity.saturating_add(1);
            return Err(StateLifecycleError::UnexpectedIdentity);
        }
        if self
            .last_transport_sequence
            .is_some_and(|previous| snapshot.transport_sequence <= previous)
            || self
                .last_snapshot_revision
                .is_some_and(|previous| snapshot.snapshot_revision <= previous)
        {
            self.stats.rejected_order = self.stats.rejected_order.saturating_add(1);
            return Err(StateLifecycleError::OutOfOrder);
        }

        let observation_timestamp = if snapshot.update_kind == UpdateKind::Observation {
            let key = (
                snapshot.source.session_id.clone(),
                snapshot.source.frame_id.clone(),
            );
            if self.last_observation.as_ref() == Some(&key) {
                self.stats.rejected_duplicate_observation =
                    self.stats.rejected_duplicate_observation.saturating_add(1);
                return Err(StateLifecycleError::DuplicateObservation);
            }
            let timestamp = timestamp_key(&snapshot.source.timestamp)?;
            if self
                .last_observation_timestamp
                .as_ref()
                .is_some_and(|previous| {
                    previous.clock == timestamp.clock
                        && previous.domain_id == timestamp.domain_id
                        && timestamp.ticks <= previous.ticks
                })
            {
                self.stats.rejected_timestamp = self.stats.rejected_timestamp.saturating_add(1);
                return Err(StateLifecycleError::StaleObservationTimestamp);
            }
            Some((key, timestamp))
        } else {
            None
        };

        // Convert and enforce retention before advancing sequence watermarks.
        // A malformed but ordered message must not poison a corrected retry.
        let new_payload = if snapshot.update_kind == UpdateKind::Observation
            && snapshot.outcome == Outcome::Success
        {
            if snapshot.balls.len() > self.config.maximum_balls {
                return Err(StateLifecycleError::BallLimitExceeded {
                    actual: snapshot.balls.len(),
                    maximum: self.config.maximum_balls,
                });
            }
            Some(
                self.adapter
                    .adapt(snapshot, table.clone(), game_type.clone())?
                    .ok_or(StateLifecycleError::MissingLocalizationPayload)?,
            )
        } else {
            None
        };

        // Commit watermarks only after every rejection check above passes.
        self.last_transport_sequence = Some(snapshot.transport_sequence);
        self.last_snapshot_revision = Some(snapshot.snapshot_revision);
        self.last_event_received_monotonic_ns = Some(received_monotonic_ns);
        if let Some((key, timestamp)) = observation_timestamp {
            self.last_observation = Some(key);
            self.last_observation_timestamp = Some(timestamp);
        }

        if snapshot.update_kind == UpdateKind::Annotation {
            self.stats.ignored_annotations = self.stats.ignored_annotations.saturating_add(1);
            return Ok(IngestOutcome::AnnotationIgnored);
        }

        if snapshot.update_kind == UpdateKind::Health || snapshot.outcome != Outcome::Success {
            self.apply_health(snapshot, received_monotonic_ns, table, game_type);
            self.stats.accepted_health_updates =
                self.stats.accepted_health_updates.saturating_add(1);
            return Ok(IngestOutcome::HealthUpdated);
        }

        let payload = new_payload.ok_or(StateLifecycleError::MissingLocalizationPayload)?;
        let mut ball_received = HashMap::with_capacity(payload.balls.len());
        let mut ball_age_offsets = HashMap::with_capacity(payload.balls.len());
        let previous = self.stored.as_ref();
        for ball in &payload.balls {
            let previous_observation = previous
                .and_then(|state| state.payload.balls.iter().find(|old| old.id == ball.id))
                .map(|old| &old.source_observation);
            let same_observation = previous_observation == Some(&ball.source_observation);
            let receipt = if same_observation {
                previous
                    .and_then(|state| state.ball_received_monotonic_ns.get(&ball.id))
                    .copied()
                    .unwrap_or(received_monotonic_ns)
            } else {
                received_monotonic_ns
            };
            ball_received.insert(ball.id.clone(), receipt);
            let age_offset = ball_age_offsets_ns
                .get(&ball.id)
                .copied()
                .or_else(|| {
                    same_observation
                        .then(|| {
                            previous
                                .and_then(|state| state.ball_age_offset_ns.get(&ball.id))
                                .copied()
                        })
                        .flatten()
                })
                .unwrap_or_default();
            ball_age_offsets.insert(ball.id.clone(), age_offset);
        }
        self.stored = Some(StoredState {
            payload,
            updated_monotonic_ns: received_monotonic_ns,
            localization_received_monotonic_ns: Some(received_monotonic_ns),
            localization_age_offset_ns,
            ball_received_monotonic_ns: ball_received,
            ball_age_offset_ns: ball_age_offsets,
            health_forced_stale: false,
            health_reason: None,
            stale_counted: false,
            expired_counted: false,
        });
        self.stats.accepted_localizations = self.stats.accepted_localizations.saturating_add(1);
        Ok(IngestOutcome::LocalizationUpdated)
    }

    /// Return one defensive current-state snapshot. Freshness is calculated at
    /// read time from the receiver's monotonic clock, never from a remote clock.
    pub fn current(
        &mut self,
        now_monotonic_ns: u128,
    ) -> Result<Option<CurrentSceneState>, StateLifecycleError> {
        let Some(stored) = self.stored.as_mut() else {
            return Ok(None);
        };
        if now_monotonic_ns < stored.updated_monotonic_ns
            || self
                .last_event_received_monotonic_ns
                .is_some_and(|last_event| now_monotonic_ns < last_event)
        {
            return Err(StateLifecycleError::ClockMovedBackwards);
        }
        let localization_age_ns = stored.localization_received_monotonic_ns.map(|receipt| {
            now_monotonic_ns
                .saturating_sub(receipt)
                .saturating_add(stored.localization_age_offset_ns)
        });
        let live_layout_trusted = self
            .latest_perception_status
            .as_ref()
            .is_some_and(|status| {
                let delivered_at = self.perception_received_monotonic_ns.unwrap_or_default();
                let elapsed_ms = now_monotonic_ns
                    .saturating_sub(delivered_at)
                    .checked_div(1_000_000)
                    .unwrap_or_default()
                    .min(u128::from(u64::MAX)) as u64;
                let age_is_current = |age_field: &str, maximum_field: &str| {
                    status
                        .get(age_field)
                        .and_then(Value::as_u64)
                        .zip(status.get(maximum_field).and_then(Value::as_u64))
                        .is_some_and(|(age, maximum)| age.saturating_add(elapsed_ms) <= maximum)
                };
                !stored.health_forced_stale
                    && status.get("layout_trusted").and_then(Value::as_bool) == Some(true)
                    && age_is_current("latest_motion_check_age_ms", "motion_evidence_max_age_ms")
            });
        let freshness = if live_layout_trusted {
            // Full detector scans are intentionally slow and periodic. Their
            // original measurement age is reported separately and must not
            // expire a layout supported by current quiet motion evidence.
            StateFreshness::Fresh
        } else {
            match localization_age_ns {
                None => StateFreshness::Stale,
                Some(age)
                    if age >= self.config.expire_after_ns && stored.payload.balls.is_empty() =>
                {
                    StateFreshness::Expired
                }
                Some(age) if age >= self.config.expire_after_ns => StateFreshness::Stale,
                Some(age) if stored.health_forced_stale || age >= self.config.stale_after_ns => {
                    StateFreshness::Stale
                }
                Some(_) => StateFreshness::Fresh,
            }
        };
        let mut payload = stored.payload.clone();
        match freshness {
            StateFreshness::Fresh => {
                payload.observation_status["freshness_status"] = json!(freshness.as_str());
                payload.observation_status["localization_age_ns"] =
                    json!(localization_age_ns.map(|age| age.to_string()));
            }
            StateFreshness::Stale => {
                mark_payload_stale(&mut payload, stored.health_reason.as_deref());
                payload.observation_status["localization_age_ns"] =
                    json!(localization_age_ns.map(|age| age.to_string()));
                if !stored.stale_counted {
                    self.stats.stale_transitions = self.stats.stale_transitions.saturating_add(1);
                    stored.stale_counted = true;
                }
            }
            StateFreshness::Expired => {
                if !stored.stale_counted {
                    self.stats.stale_transitions = self.stats.stale_transitions.saturating_add(1);
                    stored.stale_counted = true;
                }
                if !stored.expired_counted {
                    self.stats.expired_transitions =
                        self.stats.expired_transitions.saturating_add(1);
                    stored.expired_counted = true;
                }
                payload.balls.clear();
                payload.observation_status["freshness_status"] = json!(freshness.as_str());
                payload.observation_status["ball_positions_status"] = json!("unavailable");
                payload.observation_status["ball_detection_status"] = json!("unavailable");
                payload.observation_status["localization_age_ns"] =
                    json!(localization_age_ns.map(|age| age.to_string()));
                remove_current_observation_fields(&mut payload.observation_status);
            }
        }
        if let (Some(perception), Some(received_at)) = (
            self.latest_perception_status.as_ref(),
            self.perception_received_monotonic_ns,
        ) {
            payload.observation_status["perception"] = refresh_perception_status(
                perception.clone(),
                now_monotonic_ns.saturating_sub(received_at),
            );
            if payload.observation_status["perception"]
                .get("layout_trusted")
                .and_then(Value::as_bool)
                != Some(true)
            {
                if payload.balls.is_empty() {
                    // Without any ball positions, an untrusted layout cannot
                    // provide useful current coordinates. Keep empty partial
                    // observations explicitly unavailable.
                    mark_payload_stale(&mut payload, None);
                } else {
                    // Layout trust governs whole-table guidance, not whether a
                    // measured center from a recent partial scan is current.
                    // The capture pipeline marks unmatched cached tracks as
                    // unobserved; preserve these per-ball flags so clients can
                    // use fresh partial centers while guidance stays invalid
                    // until a clean layout scan restores trust.
                    payload.observation_status["layout_trusted"] = json!(false);
                    payload.observation_status["guidance_valid"] = json!(false);
                }
            }
        }
        let ball_age_ns = if freshness == StateFreshness::Expired {
            HashMap::new()
        } else {
            stored
                .ball_received_monotonic_ns
                .iter()
                .map(|(id, receipt)| {
                    (
                        id.clone(),
                        now_monotonic_ns.saturating_sub(*receipt).saturating_add(
                            stored
                                .ball_age_offset_ns
                                .get(id)
                                .copied()
                                .unwrap_or_default(),
                        ),
                    )
                })
                .collect()
        };
        Ok(Some(CurrentSceneState {
            payload,
            epoch: self.epoch,
            updated_monotonic_ns: stored.updated_monotonic_ns,
            localization_received_monotonic_ns: stored.localization_received_monotonic_ns,
            localization_age_ns,
            ball_age_ns,
            freshness,
        }))
    }

    fn apply_health(
        &mut self,
        snapshot: &SceneSnapshotV1,
        received_monotonic_ns: u128,
        table: Option<Value>,
        game_type: Option<String>,
    ) {
        if let Some(stored) = &mut self.stored {
            stored.updated_monotonic_ns = received_monotonic_ns;
            stored.health_forced_stale = true;
            stored.health_reason = snapshot.reason.clone();
            mark_payload_stale(&mut stored.payload, snapshot.reason.as_deref());
            stored.payload.observation_status["ball_detection_status"] = json!(snapshot.outcome);
            stored.payload.observation_status["health_source_frame_id"] =
                json!(snapshot.source.frame_id);
            stored.payload.observation_status["health_transport_sequence"] =
                json!(snapshot.transport_sequence);
            stored.payload.observation_status["health_snapshot_revision"] =
                json!(snapshot.snapshot_revision);
        } else if snapshot.update_kind == UpdateKind::Health || snapshot.outcome != Outcome::Success
        {
            let mut status = json!({
                "phase":"rust_scene_health",
                "engine_instance_id":snapshot.engine_instance_id,
                "source_session_id":snapshot.source.session_id,
                "health_source_frame_id":snapshot.source.frame_id,
                "health_transport_sequence":snapshot.transport_sequence,
                "health_snapshot_revision":snapshot.snapshot_revision,
                "ball_positions_status":"unavailable",
                "ball_detection_status":snapshot.outcome,
                "health_reason":snapshot.reason,
                "freshness_status":"stale"
            });
            status["configuration_revision"] = json!(snapshot.revisions.configuration);
            self.stored = Some(StoredState {
                payload: SceneStatePayload {
                    balls: Vec::new(),
                    observation_status: status,
                    table,
                    game_type,
                },
                updated_monotonic_ns: received_monotonic_ns,
                localization_received_monotonic_ns: None,
                localization_age_offset_ns: 0,
                ball_received_monotonic_ns: HashMap::new(),
                ball_age_offset_ns: HashMap::new(),
                health_forced_stale: true,
                health_reason: snapshot.reason.clone(),
                stale_counted: false,
                expired_counted: false,
            });
        }
    }
}

fn timestamp_key(timestamp: &Timestamp) -> Result<ObservationTimestamp, StateLifecycleError> {
    let ticks = timestamp
        .ticks
        .parse::<u128>()
        .map_err(|_| StateLifecycleError::InvalidSourceTimestamp)?;
    Ok(ObservationTimestamp {
        clock: timestamp.clock,
        domain_id: timestamp.domain_id.clone(),
        ticks,
    })
}

fn mark_payload_stale(payload: &mut SceneStatePayload, reason: Option<&str>) {
    for ball in &mut payload.balls {
        ball.is_observed = false;
        // A native contour describes one accepted source observation. Cached
        // position visibility must never promote that old contour to current
        // frame geometry after expiry or health degradation.
        ball.outline = None;
        ball.confidence = 0.0;
        ball.confidence_kind = "unavailable".to_owned();
    }
    payload.observation_status["freshness_status"] = json!("stale");
    payload.observation_status["ball_positions_status"] = if payload.balls.is_empty() {
        json!("unavailable")
    } else {
        json!("cached")
    };
    if reason.is_none() {
        payload.observation_status["ball_detection_status"] = json!("stale");
    }
    if let Some(reason) = reason {
        payload.observation_status["health_reason"] = json!(reason);
    }
    remove_current_observation_fields(&mut payload.observation_status);
}

fn refresh_perception_status(mut status: Value, age_since_delivery_ns: u128) -> Value {
    let Some(object) = status.as_object_mut() else {
        return Value::Null;
    };
    let age_ms = (age_since_delivery_ns / 1_000_000).min(u128::from(u64::MAX)) as u64;
    object.insert("delivery_age_ms".to_owned(), json!(age_ms));
    for (age_field, maximum_field) in [
        ("latest_motion_check_age_ms", "motion_evidence_max_age_ms"),
        (
            "last_ball_observation_age_ms",
            "layout_verification_max_age_ms",
        ),
    ] {
        if let (Some(current_age), Some(maximum)) = (
            object.get(age_field).and_then(Value::as_u64),
            object.get(maximum_field).and_then(Value::as_u64),
        ) {
            object.insert(
                age_field.to_owned(),
                json!(current_age.saturating_add(age_ms)),
            );
            if age_field == "latest_motion_check_age_ms"
                && current_age.saturating_add(age_ms) > maximum
            {
                object.insert("layout_trusted".to_owned(), json!(false));
                object.insert("empty_layout_verified".to_owned(), json!(false));
                object.insert("guidance_valid".to_owned(), json!(false));
                object.insert("phase".to_owned(), json!("recovering"));
                object.insert(
                    "trust_reason".to_owned(),
                    json!(match age_field {
                        "latest_motion_check_age_ms" => "motion_evidence_expired",
                        _ => "layout_verification_expired",
                    }),
                );
            }
        }
    }
    if let (Some(current_age), Some(maximum)) = (
        object.get("cue_evidence_age_ms").and_then(Value::as_u64),
        object
            .get("cue_evidence_max_age_ms")
            .and_then(Value::as_u64),
    ) {
        let cue_age = current_age.saturating_add(age_ms);
        object.insert("cue_evidence_age_ms".to_owned(), json!(cue_age));
        if cue_age > maximum {
            object.insert("guidance_valid".to_owned(), json!(false));
            if object.get("layout_trusted").and_then(Value::as_bool) == Some(true) {
                object.insert("phase".to_owned(), json!("pre_shot"));
            }
        }
    }
    if object.get("empty_layout_verified").and_then(Value::as_bool) == Some(true) {
        let evidence_is_fresh = |age_field: &str, maximum_field: &str| {
            matches!(
                (
                    object.get(age_field).and_then(Value::as_u64),
                    object.get(maximum_field).and_then(Value::as_u64)
                ),
                (Some(age), Some(maximum)) if age <= maximum
            )
        };
        if object.get("layout_trusted").and_then(Value::as_bool) != Some(true)
            || !evidence_is_fresh("latest_motion_check_age_ms", "motion_evidence_max_age_ms")
        {
            object.insert("empty_layout_verified".to_owned(), json!(false));
        }
    }
    status
}

fn remove_current_observation_fields(status: &mut Value) {
    if let Some(object) = status.as_object_mut() {
        for field in [
            "source_frame_id",
            "source_timestamp",
            "source_timestamp_unit",
            "source_timestamp_kind",
            "source_timestamp_clock",
            "source_timestamp_domain_id",
            "source_timestamp_is_exposure",
            "published_at",
            "frame_publication_age_seconds",
            "frame_publication_age_status",
            "frame_age_basis",
            "snapshot_revision",
            "transport_sequence",
        ] {
            object.remove(field);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_scan_age_does_not_clear_trust_while_motion_evidence_is_current() {
        let status = json!({
            "phase": "pre_shot",
            "layout_trusted": true,
            "empty_layout_verified": true,
            "guidance_valid": false,
            "trust_reason": "trusted",
            "last_ball_observation_age_ms": 1_499,
            "latest_motion_check_age_ms": 0,
            "cue_evidence_age_ms": null,
            "motion_evidence_max_age_ms": 200,
            "cue_evidence_max_age_ms": 250,
            "layout_verification_max_age_ms": 1_500
        });

        let refreshed = refresh_perception_status(status, 2_000_000);
        assert_eq!(refreshed["layout_trusted"], true);
        assert_eq!(refreshed["empty_layout_verified"], true);
        assert_eq!(refreshed["phase"], "pre_shot");
        assert_eq!(refreshed["last_ball_observation_age_ms"], 1_501);
    }

    #[test]
    fn verified_empty_status_fails_closed_without_motion_evidence() {
        let status = json!({
            "layout_trusted": true,
            "empty_layout_verified": true,
            "guidance_valid": false,
            "last_ball_observation_age_ms": 0,
            "latest_motion_check_age_ms": null,
            "motion_evidence_max_age_ms": 200,
            "layout_verification_max_age_ms": 1_500
        });

        let refreshed = refresh_perception_status(status, 0);
        assert_eq!(refreshed["layout_trusted"], true);
        assert_eq!(refreshed["empty_layout_verified"], false);
    }
}

#[derive(Debug)]
pub enum StateLifecycleError {
    PublicState(PublicStateError),
    InvalidConfiguration,
    InvalidIdentity,
    SourceAlreadyActive,
    EpochExhausted,
    SourceNotActivated,
    UnexpectedIdentity,
    OutOfOrder,
    DuplicateObservation,
    StaleObservationTimestamp,
    InvalidSourceTimestamp,
    ClockMovedBackwards,
    InvalidPerceptionStatus,
    MissingLocalizationPayload,
    BallLimitExceeded { actual: usize, maximum: usize },
}
impl fmt::Display for StateLifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PublicState(error) => error.fmt(f),
            Self::InvalidConfiguration => f.write_str("invalid state lifecycle configuration"),
            Self::InvalidIdentity => f.write_str("source identity must be non-empty and bounded"),
            Self::SourceAlreadyActive => {
                f.write_str("a source restart requires a new engine or session identity")
            }
            Self::EpochExhausted => f.write_str("source epoch counter exhausted"),
            Self::SourceNotActivated => f.write_str("source identity was not activated"),
            Self::UnexpectedIdentity => f.write_str("scene identity differs from active source"),
            Self::OutOfOrder => f.write_str("scene sequence or revision is duplicate/out of order"),
            Self::DuplicateObservation => f.write_str("source observation was already accepted"),
            Self::StaleObservationTimestamp => {
                f.write_str("source observation timestamp did not advance")
            }
            Self::InvalidSourceTimestamp => {
                f.write_str("source timestamp is not a decimal nanosecond value")
            }
            Self::ClockMovedBackwards => f.write_str("local monotonic clock moved backwards"),
            Self::InvalidPerceptionStatus => f.write_str("perception status must be a JSON object"),
            Self::MissingLocalizationPayload => {
                f.write_str("successful observation produced no state payload")
            }
            Self::BallLimitExceeded { actual, maximum } => write!(
                f,
                "scene has {actual} balls; configured maximum is {maximum}"
            ),
        }
    }
}
impl Error for StateLifecycleError {}
impl From<PublicStateError> for StateLifecycleError {
    fn from(error: PublicStateError) -> Self {
        Self::PublicState(error)
    }
}
