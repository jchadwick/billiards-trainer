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
        self.stored = None;
        Ok(self.epoch)
    }

    pub fn stats(&self) -> StateLifecycleStats {
        self.stats
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
        let freshness = match localization_age_ns {
            None => StateFreshness::Stale,
            Some(age) if age >= self.config.expire_after_ns => StateFreshness::Expired,
            Some(age) if stored.health_forced_stale || age >= self.config.stale_after_ns => {
                StateFreshness::Stale
            }
            Some(_) => StateFreshness::Fresh,
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
