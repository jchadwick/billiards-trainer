//! Source-time association from complete detections to complete scene snapshots.
//!
//! The tracker keeps detector centers as measurements. A bounded constant-
//! velocity estimate is used only to choose associations; it is never
//! published as if it were a current observed location. Configuration is
//! supplied by the caller so the experimental thresholds are visible and can
//! later be tied to revisioned application settings.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Duration;

use crate::calibration::ValidatedTableCalibrationV1;
use crate::detection_batch::{
    BatchOutcome, BatchTimestamp, ClockKind as BatchClockKind, DetectionBatchV1,
    ImageSpace as BatchImageSpace, TimeUnit as BatchTimeUnit,
    TimestampStage as BatchTimestampStage,
};
use crate::geometry::{
    calibrate_detection_batch, AppliedCalibrationContext, CalibratedDetectionBatch,
};
use crate::{
    BallIdentity, BallState, BallStatus, CalibrationState, CueState, CueStatus, ImageSpace,
    Measurement, Outcome, Point2d, PositionUncertainty, Revisions, SceneSnapshotV1, SourceFrame,
    TimeUnit, Timestamp, TimestampStage, UncertaintyStatus, UpdateKind,
    SCENE_SNAPSHOT_SCHEMA_VERSION,
};

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Tunable temporal-association limits for the source-image prototype.
#[derive(Debug, Clone)]
pub struct TrackingConfigV1 {
    ball_class_id: u32,
    cue_class_id: Option<u32>,
    camera_profile_id: String,
    association_gate_px: f64,
    association_prediction_horizon: Duration,
    lost_after: Duration,
    maximum_track_age: Duration,
    maximum_active_tracks: usize,
    recent_observation_capacity: usize,
    configuration_revision: u64,
    rectification_revision: u64,
}

impl TrackingConfigV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ball_class_id: u32,
        cue_class_id: Option<u32>,
        camera_profile_id: impl Into<String>,
        association_gate_px: f64,
        association_prediction_horizon: Duration,
        lost_after: Duration,
        maximum_track_age: Duration,
        maximum_active_tracks: usize,
        recent_observation_capacity: usize,
        configuration_revision: u64,
        rectification_revision: u64,
    ) -> Result<Self, TrackingError> {
        if cue_class_id == Some(ball_class_id) {
            return Err(TrackingError::InvalidConfiguration(
                "cue and ball class IDs must differ",
            ));
        }
        let camera_profile_id = camera_profile_id.into();
        if camera_profile_id.is_empty()
            || camera_profile_id.chars().count() > crate::detection_batch::MAX_IDENTIFIER_CHARACTERS
            || camera_profile_id.chars().any(char::is_control)
        {
            return Err(TrackingError::InvalidConfiguration(
                "camera profile ID must be a bounded non-empty identifier",
            ));
        }
        if !association_gate_px.is_finite() || association_gate_px <= 0.0 {
            return Err(TrackingError::InvalidConfiguration(
                "association gate must be finite and positive",
            ));
        }
        if lost_after.is_zero() || maximum_track_age <= lost_after {
            return Err(TrackingError::InvalidConfiguration(
                "maximum track age must be greater than a positive lost interval",
            ));
        }
        if maximum_active_tracks == 0 || recent_observation_capacity == 0 {
            return Err(TrackingError::InvalidConfiguration(
                "track and recent-observation capacities must be positive",
            ));
        }
        if configuration_revision > MAX_SAFE_JSON_INTEGER
            || rectification_revision > MAX_SAFE_JSON_INTEGER
        {
            return Err(TrackingError::InvalidConfiguration(
                "configuration revisions must be safe JSON integers",
            ));
        }
        let unmatched_cost = association_gate_px * (maximum_active_tracks as f64 + 1.0) + 1.0;
        if !unmatched_cost.is_finite() {
            return Err(TrackingError::InvalidConfiguration(
                "association gate is too large for the configured track capacity",
            ));
        }

        Ok(Self {
            ball_class_id,
            cue_class_id,
            camera_profile_id,
            association_gate_px,
            association_prediction_horizon,
            lost_after,
            maximum_track_age,
            maximum_active_tracks,
            recent_observation_capacity,
            configuration_revision,
            rectification_revision,
        })
    }

    pub fn ball_class_id(&self) -> u32 {
        self.ball_class_id
    }
    pub fn cue_class_id(&self) -> Option<u32> {
        self.cue_class_id
    }
    pub fn camera_profile_id(&self) -> &str {
        &self.camera_profile_id
    }
    pub fn association_gate_px(&self) -> f64 {
        self.association_gate_px
    }
    pub fn association_prediction_horizon(&self) -> Duration {
        self.association_prediction_horizon
    }
    pub fn lost_after(&self) -> Duration {
        self.lost_after
    }
    pub fn maximum_track_age(&self) -> Duration {
        self.maximum_track_age
    }
    pub fn maximum_active_tracks(&self) -> usize {
        self.maximum_active_tracks
    }
    pub fn recent_observation_capacity(&self) -> usize {
        self.recent_observation_capacity
    }
    pub fn configuration_revision(&self) -> u64 {
        self.configuration_revision
    }
    pub fn rectification_revision(&self) -> u64 {
        self.rectification_revision
    }
}

/// Fail-closed conditions that prevent a batch from becoming a fresh scene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackingError {
    InvalidConfiguration(&'static str),
    InvalidBatch(String),
    UnsuccessfulBatch(BatchOutcome),
    MissingSourceTimestamp,
    InvalidSourceTimestamp,
    InvalidPublicationTimestamp,
    NonIncreasingSourceTime,
    NonIncreasingBatchSequence,
    DuplicateSourceFrame,
    MissingBallClass,
    MissingCueClass,
    AppliedCalibrationContextMismatch,
    PrecalibratedBatchMismatch(&'static str),
    ActiveTrackCapacityExceeded,
    CounterExhausted,
    InvalidSnapshot(String),
}

impl Display for TrackingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "source-time tracking error: {self:?}")
    }
}

impl Error for TrackingError {}

/// A complete scene plus old association entries evicted to fit the current
/// frame under the configured track-history bound.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneTrackingResult {
    pub snapshot: SceneSnapshotV1,
    pub association_capacity_evictions: usize,
}

#[derive(Debug, Clone)]
struct Track {
    id: String,
    previous_center: Option<Point2d>,
    previous_timestamp_ns: Option<u128>,
    center: Point2d,
    last_timestamp_ns: u128,
    last_observation: crate::LastObservation,
    measurement: Measurement,
    calibration_revision: Option<u64>,
}

/// Calibration acknowledged by the replay/runtime boundary for one source frame.
#[derive(Clone, Copy)]
pub struct AppliedTrackingCalibration<'a> {
    pub calibration: &'a ValidatedTableCalibrationV1,
    pub context: AppliedCalibrationContext<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceIdentity {
    engine_instance_id: String,
    session_id: String,
    model_sha256: String,
    camera_profile_id: String,
    image_space: ImageSpace,
    width_px: u32,
    height_px: u32,
    clock_kind: crate::ClockKind,
    clock_domain_id: String,
    source_timestamp_stage: TimestampStage,
    configuration_revision: u64,
    rectification_revision: u64,
}

/// Stateful association for one source stream. It can be reset and reused
/// safely; source/session/model/geometry changes invalidate active tracks.
#[derive(Debug, Clone)]
pub struct SceneTrackerV1 {
    config: TrackingConfigV1,
    tracks: Vec<Track>,
    next_track_id: u64,
    snapshot_revision: u64,
    transport_sequence: u64,
    identity: Option<SourceIdentity>,
    last_source_timestamp_ns: Option<u128>,
    last_batch_sequence: Option<u64>,
    recent_frame_ids: VecDeque<String>,
    recent_frame_id_set: HashSet<String>,
}

impl SceneTrackerV1 {
    pub fn new(config: TrackingConfigV1) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_track_id: 1,
            snapshot_revision: 0,
            transport_sequence: 0,
            identity: None,
            last_source_timestamp_ns: None,
            last_batch_sequence: None,
            recent_frame_ids: VecDeque::new(),
            recent_frame_id_set: HashSet::new(),
        }
    }

    /// Replace applied tracking settings without recycling track IDs or scene
    /// sequence numbers. The next accepted frame starts a fresh association
    /// epoch under the new settings.
    pub fn reconfigure(&mut self, config: TrackingConfigV1) {
        self.tracks.clear();
        self.config = config;
    }

    /// Process one unique successful source observation transactionally.
    /// Errors leave tracker state untouched, so failed inference cannot age or
    /// erase the last valid scene.
    pub fn process(
        &mut self,
        batch: &DetectionBatchV1,
        published_at: Timestamp,
    ) -> Result<SceneSnapshotV1, TrackingError> {
        self.process_with_calibration(batch, published_at, None)
    }

    /// Process one observation and, when configured, attach physical table
    /// coordinates only when the complete frame passes calibration checks.
    /// Invalid calibration degrades the scene without discarding image-space
    /// detections or advancing stale physical coordinates.
    pub fn process_with_calibration(
        &mut self,
        batch: &DetectionBatchV1,
        published_at: Timestamp,
        calibration: Option<AppliedTrackingCalibration<'_>>,
    ) -> Result<SceneSnapshotV1, TrackingError> {
        if calibration.is_some_and(|applied| {
            applied.context.camera_profile_id != self.config.camera_profile_id
                || applied.context.rectification_revision != self.config.rectification_revision
        }) {
            return Err(TrackingError::AppliedCalibrationContextMismatch);
        }
        let calibrated = calibration.map(|applied| {
            let result = calibrate_detection_batch(batch, applied.calibration, applied.context);
            (applied.context.calibration_revision, result)
        });
        self.process_with_precalibrated_batch(batch, published_at, calibrated)
    }

    /// Apply an already computed frame calibration result. This lets replay
    /// report calibration and build a scene from one geometry pass.
    pub fn process_with_precalibrated_batch(
        &mut self,
        batch: &DetectionBatchV1,
        published_at: Timestamp,
        calibrated: Option<(
            u64,
            Result<CalibratedDetectionBatch, crate::geometry::GeometryError>,
        )>,
    ) -> Result<SceneSnapshotV1, TrackingError> {
        self.process_with_precalibrated_batch_report(batch, published_at, calibrated)
            .map(|result| result.snapshot)
    }

    /// Process one observation and report old associations evicted to keep all
    /// eligible current detections in the emitted scene.
    pub fn process_with_precalibrated_batch_report(
        &mut self,
        batch: &DetectionBatchV1,
        published_at: Timestamp,
        calibrated: Option<(
            u64,
            Result<CalibratedDetectionBatch, crate::geometry::GeometryError>,
        )>,
    ) -> Result<SceneTrackingResult, TrackingError> {
        if let Some((revision, Ok(calibrated_batch))) = calibrated.as_ref() {
            validate_precalibrated_batch(
                batch,
                *revision,
                calibrated_batch,
                &self.config.camera_profile_id,
                self.config.rectification_revision,
            )?;
        }
        let mut next = self.clone();
        let (snapshot, association_capacity_evictions) =
            next.process_inner(batch, published_at, calibrated)?;
        *self = next;
        Ok(SceneTrackingResult {
            snapshot,
            association_capacity_evictions,
        })
    }

    fn process_inner(
        &mut self,
        batch: &DetectionBatchV1,
        published_at: Timestamp,
        calibrated: Option<(
            u64,
            Result<CalibratedDetectionBatch, crate::geometry::GeometryError>,
        )>,
    ) -> Result<(SceneSnapshotV1, usize), TrackingError> {
        batch
            .validate()
            .map_err(|error| TrackingError::InvalidBatch(error.to_string()))?;
        if batch.outcome != BatchOutcome::Success {
            return Err(TrackingError::UnsuccessfulBatch(batch.outcome));
        }
        if published_at.stage != TimestampStage::SnapshotPublication {
            return Err(TrackingError::InvalidPublicationTimestamp);
        }
        if !batch
            .model
            .classes
            .iter()
            .any(|class| class.class_id == self.config.ball_class_id)
        {
            return Err(TrackingError::MissingBallClass);
        }
        if self.config.cue_class_id.is_some_and(|cue_class_id| {
            !batch
                .model
                .classes
                .iter()
                .any(|class| class.class_id == cue_class_id)
        }) {
            return Err(TrackingError::MissingCueClass);
        }

        let source_timestamp = batch
            .source
            .timestamp
            .as_ref()
            .ok_or(TrackingError::MissingSourceTimestamp)?;
        let source_timestamp_ns = parse_timestamp_ticks(source_timestamp)?;
        let mapped_source_timestamp = map_source_timestamp(source_timestamp)?;
        let source_image_space = map_image_space(batch.image_space);
        let source_clock_kind = map_clock_kind(source_timestamp.clock);
        let identity = SourceIdentity {
            engine_instance_id: batch.engine_instance_id.clone(),
            session_id: batch.source.session_id.clone(),
            model_sha256: batch.model.artifact_sha256.clone(),
            camera_profile_id: self.config.camera_profile_id.clone(),
            image_space: source_image_space,
            width_px: batch.source.width_px,
            height_px: batch.source.height_px,
            clock_kind: source_clock_kind,
            clock_domain_id: source_timestamp.domain_id.clone(),
            source_timestamp_stage: mapped_source_timestamp.stage,
            configuration_revision: self.config.configuration_revision,
            rectification_revision: self.config.rectification_revision,
        };

        let new_engine = self
            .identity
            .as_ref()
            .is_some_and(|old| old.engine_instance_id != identity.engine_instance_id);
        let new_session = self.identity.as_ref().is_some_and(|old| {
            old.engine_instance_id == identity.engine_instance_id
                && old.session_id != identity.session_id
        });
        if new_engine {
            self.reset_engine_instance();
        } else if new_session {
            self.reset_source_session();
        }

        let previous_identity = self.identity.as_ref();
        let same_session = previous_identity.is_some_and(|old| {
            old.engine_instance_id == identity.engine_instance_id
                && old.session_id == identity.session_id
        });
        if same_session {
            if self.recent_frame_id_set.contains(&batch.source.frame_id) {
                return Err(TrackingError::DuplicateSourceFrame);
            }
            if previous_identity.is_some_and(|old| {
                old.clock_kind == identity.clock_kind
                    && old.clock_domain_id == identity.clock_domain_id
            }) {
                if self
                    .last_source_timestamp_ns
                    .is_some_and(|last| source_timestamp_ns <= last)
                {
                    return Err(TrackingError::NonIncreasingSourceTime);
                }
                if self
                    .last_batch_sequence
                    .is_some_and(|last| batch.batch_sequence <= last)
                {
                    return Err(TrackingError::NonIncreasingBatchSequence);
                }
            }
            if previous_identity != Some(&identity) {
                // Geometry/model/configuration/clock changes make old spatial
                // association unsafe but do not recycle IDs in this session.
                self.tracks.clear();
                self.last_source_timestamp_ns = None;
                self.last_batch_sequence = None;
            }
        }

        let now_ns = source_timestamp_ns;
        let max_age_ns = self.config.maximum_track_age.as_nanos();
        self.tracks
            .retain(|track| now_ns.saturating_sub(track.last_timestamp_ns) < max_age_ns);

        let ball_detections = batch
            .detections
            .iter()
            .filter(|detection| detection.class_id == self.config.ball_class_id)
            .collect::<Vec<_>>();
        let mut assignments = self.assign_detections(&ball_detections, now_ns)?;
        let unmatched_count = ball_detections.len().saturating_sub(assignments.len());
        let projected_track_count = self
            .tracks
            .len()
            .checked_add(unmatched_count)
            .ok_or(TrackingError::ActiveTrackCapacityExceeded)?;
        let association_capacity_evictions =
            projected_track_count.saturating_sub(self.config.maximum_active_tracks);
        if association_capacity_evictions > 0 {
            let assigned_track_indices = assignments
                .iter()
                .map(|(track_index, _)| *track_index)
                .collect::<HashSet<_>>();
            let mut eviction_candidates = (0..self.tracks.len())
                .filter(|track_index| !assigned_track_indices.contains(track_index))
                .collect::<Vec<_>>();
            eviction_candidates.sort_by(|left, right| {
                self.tracks[*left]
                    .last_timestamp_ns
                    .cmp(&self.tracks[*right].last_timestamp_ns)
                    .then_with(|| self.tracks[*left].id.cmp(&self.tracks[*right].id))
            });
            if eviction_candidates.len() < association_capacity_evictions {
                // The current frame itself cannot fit in the configured bound.
                // Live construction rejects this mismatch before acquisition;
                // malformed replay batches remain an explicit transactional error.
                return Err(TrackingError::ActiveTrackCapacityExceeded);
            }

            let evicted_track_indices = eviction_candidates
                .into_iter()
                .take(association_capacity_evictions)
                .collect::<HashSet<_>>();
            let old_tracks = std::mem::take(&mut self.tracks);
            let mut old_to_new_indices = vec![None; old_tracks.len()];
            let mut retained_tracks = Vec::with_capacity(
                old_tracks
                    .len()
                    .saturating_sub(association_capacity_evictions),
            );
            for (old_index, track) in old_tracks.into_iter().enumerate() {
                if evicted_track_indices.contains(&old_index) {
                    continue;
                }
                old_to_new_indices[old_index] = Some(retained_tracks.len());
                retained_tracks.push(track);
            }
            self.tracks = retained_tracks;
            for (track_index, _) in &mut assignments {
                *track_index = old_to_new_indices[*track_index]
                    .ok_or(TrackingError::ActiveTrackCapacityExceeded)?;
            }
        }
        let mut matched_track_indices = HashSet::with_capacity(assignments.len());
        let mut matched_detection_indices = HashSet::with_capacity(assignments.len());
        let last_observation = crate::LastObservation {
            session_id: batch.source.session_id.clone(),
            frame_id: batch.source.frame_id.clone(),
            timestamp: mapped_source_timestamp.clone(),
        };

        let (calibration_state, calibration_revision, calibrated_centers, calibration_reason) =
            match calibrated {
                None => (CalibrationState::Unavailable, None, HashMap::new(), None),
                Some((revision, Ok(calibrated_batch))) => (
                    CalibrationState::Valid,
                    Some(revision),
                    calibrated_batch
                        .detections
                        .into_iter()
                        .map(|detection| {
                            (
                                detection.detection_index,
                                Point2d {
                                    x: detection.center_table_mm.x_mm,
                                    y: detection.center_table_mm.y_mm,
                                },
                            )
                        })
                        .collect(),
                    None,
                ),
                Some((revision, Err(error))) => (
                    CalibrationState::Invalid,
                    Some(revision),
                    HashMap::new(),
                    Some(error.to_string()),
                ),
            };
        let detection_table_centers = calibrated_centers;

        for (track_index, detection_index) in assignments {
            let detection = ball_detections[detection_index];
            let measurement = image_measurement(
                detection,
                source_image_space,
                batch.source.width_px,
                batch.source.height_px,
                detection_table_centers
                    .get(&detection.detection_index)
                    .copied(),
            );
            let track = &mut self.tracks[track_index];
            track.previous_center = Some(track.center);
            track.previous_timestamp_ns = Some(track.last_timestamp_ns);
            track.center = Point2d {
                x: detection.center_image_px.x,
                y: detection.center_image_px.y,
            };
            track.last_timestamp_ns = now_ns;
            track.last_observation = last_observation.clone();
            track.measurement = measurement;
            track.calibration_revision = calibration_revision.filter(|_| {
                calibration_state == CalibrationState::Valid
                    && detection_table_centers.contains_key(&detection.detection_index)
            });
            matched_track_indices.insert(track_index);
            matched_detection_indices.insert(detection_index);
        }

        for (detection_index, detection) in ball_detections.iter().enumerate() {
            if matched_detection_indices.contains(&detection_index) {
                continue;
            }
            let id = self.allocate_track_id()?;
            let center = Point2d {
                x: detection.center_image_px.x,
                y: detection.center_image_px.y,
            };
            self.tracks.push(Track {
                id,
                previous_center: None,
                previous_timestamp_ns: None,
                center,
                last_timestamp_ns: now_ns,
                last_observation: last_observation.clone(),
                measurement: image_measurement(
                    detection,
                    source_image_space,
                    batch.source.width_px,
                    batch.source.height_px,
                    detection_table_centers
                        .get(&detection.detection_index)
                        .copied(),
                ),
                calibration_revision: calibration_revision.filter(|_| {
                    calibration_state == CalibrationState::Valid
                        && detection_table_centers.contains_key(&detection.detection_index)
                }),
            });
            matched_track_indices.insert(self.tracks.len() - 1);
        }

        let lost_after_ns = self.config.lost_after.as_nanos();
        let mut balls = Vec::with_capacity(self.tracks.len());
        for index in 0..self.tracks.len() {
            if calibration_state != CalibrationState::Valid
                || self.tracks[index].calibration_revision != calibration_revision
            {
                // A prior table point is no longer justified when calibration
                // is unavailable, invalid, or changed since this observation.
                let track = &mut self.tracks[index];
                track.measurement.table_center_mm = None;
                track.measurement.table_position_uncertainty = unavailable_uncertainty();
                track.calibration_revision = None;
            }
            let track = &self.tracks[index];
            let observed_now = matched_track_indices.contains(&index);
            let age_ns = now_ns.saturating_sub(track.last_timestamp_ns);
            let status = if observed_now {
                BallStatus::Observed
            } else if age_ns >= lost_after_ns {
                BallStatus::Lost
            } else {
                BallStatus::Occluded
            };
            balls.push(BallState {
                track_id: track.id.clone(),
                status,
                last_observation: track.last_observation.clone(),
                measurement: Some(track.measurement.clone()),
                estimate: None,
                identity: BallIdentity {
                    number: None,
                    confidence: None,
                    evidence: Vec::new(),
                },
            });
        }

        let cue = make_cue_state(
            batch,
            self.config.cue_class_id,
            source_image_space,
            &last_observation,
        );
        self.snapshot_revision = self
            .snapshot_revision
            .checked_add(1)
            .filter(|value| *value <= MAX_SAFE_JSON_INTEGER)
            .ok_or(TrackingError::CounterExhausted)?;
        self.transport_sequence = self
            .transport_sequence
            .checked_add(1)
            .filter(|value| *value <= MAX_SAFE_JSON_INTEGER)
            .ok_or(TrackingError::CounterExhausted)?;

        let mut degradation_reasons = Vec::with_capacity(2);
        if let Some(reason) = calibration_reason {
            degradation_reasons.push(reason);
        }
        if association_capacity_evictions > 0 {
            degradation_reasons.push("association_capacity_pressure".to_owned());
        }
        let snapshot = SceneSnapshotV1 {
            schema_version: SCENE_SNAPSHOT_SCHEMA_VERSION,
            engine_instance_id: batch.engine_instance_id.clone(),
            source: SourceFrame {
                session_id: batch.source.session_id.clone(),
                frame_id: batch.source.frame_id.clone(),
                timestamp: mapped_source_timestamp,
                width_px: batch.source.width_px,
                height_px: batch.source.height_px,
            },
            snapshot_revision: self.snapshot_revision,
            transport_sequence: self.transport_sequence,
            update_kind: UpdateKind::Observation,
            published_at,
            revisions: Revisions {
                configuration: self.config.configuration_revision,
                rectification: self.config.rectification_revision,
                calibration: calibration_revision,
                calibration_state,
            },
            outcome: if degradation_reasons.is_empty() {
                Outcome::Success
            } else {
                Outcome::Degraded
            },
            reason: (!degradation_reasons.is_empty()).then(|| degradation_reasons.join("; ")),
            cue,
            balls,
        };
        snapshot
            .validate()
            .map_err(|error| TrackingError::InvalidSnapshot(error.to_string()))?;

        self.identity = Some(identity);
        self.last_source_timestamp_ns = Some(source_timestamp_ns);
        self.last_batch_sequence = Some(batch.batch_sequence);
        self.remember_frame_id(batch.source.frame_id.clone());
        Ok((snapshot, association_capacity_evictions))
    }

    fn assign_detections(
        &self,
        detections: &[&crate::detection_batch::Detection],
        now_ns: u128,
    ) -> Result<Vec<(usize, usize)>, TrackingError> {
        if self.tracks.is_empty() || detections.is_empty() {
            return Ok(Vec::new());
        }
        let row_count = self.tracks.len();
        let column_count = detections.len() + row_count;
        let gate = self.config.association_gate_px;
        let unmatched_cost = gate * (row_count as f64 + 1.0) + 1.0;
        let invalid_cost = unmatched_cost + gate + 1.0;
        if !unmatched_cost.is_finite() || !invalid_cost.is_finite() {
            return Err(TrackingError::InvalidConfiguration(
                "association costs exceed finite numeric range",
            ));
        }

        let mut costs = vec![vec![unmatched_cost; column_count]; row_count];
        for (track_index, track) in self.tracks.iter().enumerate() {
            let predicted = self.predicted_association_center(track, now_ns);
            for (detection_index, detection) in detections.iter().enumerate() {
                let distance = euclidean_distance(
                    predicted,
                    Point2d {
                        x: detection.center_image_px.x,
                        y: detection.center_image_px.y,
                    },
                );
                costs[track_index][detection_index] = if distance <= gate {
                    distance
                } else {
                    invalid_cost
                };
            }
        }
        let assignment = minimum_cost_assignment(&costs);
        Ok(assignment
            .into_iter()
            .enumerate()
            .filter_map(|(track_index, detection_index)| {
                (detection_index < detections.len()
                    && costs[track_index][detection_index] < unmatched_cost)
                    .then_some((track_index, detection_index))
            })
            .collect())
    }

    fn predicted_association_center(&self, track: &Track, now_ns: u128) -> Point2d {
        let (Some(previous_center), Some(previous_timestamp_ns)) =
            (track.previous_center, track.previous_timestamp_ns)
        else {
            return track.center;
        };
        if track.last_timestamp_ns <= previous_timestamp_ns {
            return track.center;
        }
        let elapsed_ns = now_ns.saturating_sub(track.last_timestamp_ns);
        let horizon_ns = self.config.association_prediction_horizon.as_nanos();
        let prediction_ns = elapsed_ns.min(horizon_ns);
        if prediction_ns == 0 {
            return track.center;
        }
        let observed_interval = (track.last_timestamp_ns - previous_timestamp_ns) as f64;
        let prediction_interval = prediction_ns as f64;
        let scale = prediction_interval / observed_interval;
        Point2d {
            x: track.center.x + (track.center.x - previous_center.x) * scale,
            y: track.center.y + (track.center.y - previous_center.y) * scale,
        }
    }

    fn allocate_track_id(&mut self) -> Result<String, TrackingError> {
        let id_value = self.next_track_id;
        if id_value > MAX_SAFE_JSON_INTEGER {
            return Err(TrackingError::CounterExhausted);
        }
        self.next_track_id = self
            .next_track_id
            .checked_add(1)
            .ok_or(TrackingError::CounterExhausted)?;
        Ok(format!("track-{id_value}"))
    }

    fn remember_frame_id(&mut self, frame_id: String) {
        self.recent_frame_id_set.insert(frame_id.clone());
        self.recent_frame_ids.push_back(frame_id);
        while self.recent_frame_ids.len() > self.config.recent_observation_capacity {
            if let Some(expired) = self.recent_frame_ids.pop_front() {
                self.recent_frame_id_set.remove(&expired);
            }
        }
    }

    fn reset_engine_instance(&mut self) {
        self.tracks.clear();
        self.next_track_id = 1;
        self.snapshot_revision = 0;
        self.transport_sequence = 0;
        self.identity = None;
        self.last_source_timestamp_ns = None;
        self.last_batch_sequence = None;
        self.recent_frame_ids.clear();
        self.recent_frame_id_set.clear();
    }

    fn reset_source_session(&mut self) {
        self.tracks.clear();
        self.next_track_id = 1;
        self.identity = None;
        self.last_source_timestamp_ns = None;
        self.last_batch_sequence = None;
        self.recent_frame_ids.clear();
        self.recent_frame_id_set.clear();
    }
}

/// A precomputed transform may be reused only when it demonstrably belongs to
/// this exact observation and applied camera/calibration context. Without this
/// check a valid-looking image detection could be paired with a stale physical
/// coordinate and silently put a ball in the wrong place on the table.
fn validate_precalibrated_batch(
    batch: &DetectionBatchV1,
    calibration_revision: u64,
    calibrated: &CalibratedDetectionBatch,
    camera_profile_id: &str,
    rectification_revision: u64,
) -> Result<(), TrackingError> {
    if calibrated.engine_instance_id != batch.engine_instance_id {
        return Err(TrackingError::PrecalibratedBatchMismatch(
            "engine_instance_id",
        ));
    }
    if calibrated.batch_sequence != batch.batch_sequence {
        return Err(TrackingError::PrecalibratedBatchMismatch("batch_sequence"));
    }
    if calibrated.source != batch.source {
        return Err(TrackingError::PrecalibratedBatchMismatch("source_frame"));
    }
    if calibrated.image_space != batch.image_space {
        return Err(TrackingError::PrecalibratedBatchMismatch("image_space"));
    }
    if calibrated.camera_profile_id != camera_profile_id {
        return Err(TrackingError::PrecalibratedBatchMismatch(
            "camera_profile_id",
        ));
    }
    if calibrated.calibration_revision != calibration_revision {
        return Err(TrackingError::PrecalibratedBatchMismatch(
            "calibration_revision",
        ));
    }
    if calibrated.rectification_revision != rectification_revision {
        return Err(TrackingError::PrecalibratedBatchMismatch(
            "rectification_revision",
        ));
    }
    if calibrated.detections.len() != batch.detections.len() {
        return Err(TrackingError::PrecalibratedBatchMismatch("detection_count"));
    }

    for (source, transformed) in batch.detections.iter().zip(&calibrated.detections) {
        if transformed.detection_index != source.detection_index
            || transformed.class_id != source.class_id
            || transformed.class_name != source.class_name
            || transformed.confidence != source.confidence
            || transformed.center_image_px != source.center_image_px
        {
            return Err(TrackingError::PrecalibratedBatchMismatch(
                "detection_identity_or_measurement",
            ));
        }
    }
    Ok(())
}

fn image_measurement(
    detection: &crate::detection_batch::Detection,
    image_space: ImageSpace,
    width_px: u32,
    height_px: u32,
    table_center_mm: Option<Point2d>,
) -> Measurement {
    Measurement {
        image_center_px: Point2d {
            x: detection.center_image_px.x,
            y: detection.center_image_px.y,
        },
        image_radius_px: bbox_half_extent(detection),
        image_space,
        image_width_px: width_px,
        image_height_px: height_px,
        table_center_mm,
        detector_confidence: detection.confidence,
        table_position_uncertainty: PositionUncertainty {
            status: UncertaintyStatus::Unavailable,
            sigma_x_mm: None,
            sigma_y_mm: None,
        },
    }
}

/// Detector boxes are frame-scoped measurements. Keep their half-width and
/// half-height with the track so the overlay can match balls at their actual
/// apparent size, including perspective and changes in source resolution.
fn bbox_half_extent(detection: &crate::detection_batch::Detection) -> Option<Point2d> {
    let bbox = detection.bbox_image_px;
    let x = (bbox.right - bbox.left) / 2.0;
    let y = (bbox.bottom - bbox.top) / 2.0;
    (x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0).then_some(Point2d { x, y })
}

fn unavailable_uncertainty() -> PositionUncertainty {
    PositionUncertainty {
        status: UncertaintyStatus::Unavailable,
        sigma_x_mm: None,
        sigma_y_mm: None,
    }
}

fn make_cue_state(
    batch: &DetectionBatchV1,
    cue_class_id: Option<u32>,
    image_space: ImageSpace,
    observation: &crate::LastObservation,
) -> CueState {
    let Some(cue_class_id) = cue_class_id else {
        return CueState {
            status: CueStatus::Unavailable,
            last_observation: None,
            image_center_px: None,
            image_space: None,
            image_width_px: None,
            image_height_px: None,
            confidence: None,
        };
    };
    let best_cue = batch
        .detections
        .iter()
        .filter(|detection| detection.class_id == cue_class_id)
        .max_by(|left, right| left.confidence.total_cmp(&right.confidence));
    if let Some(detection) = best_cue {
        CueState {
            status: CueStatus::Observed,
            last_observation: Some(observation.clone()),
            image_center_px: Some(Point2d {
                x: detection.center_image_px.x,
                y: detection.center_image_px.y,
            }),
            image_space: Some(image_space),
            image_width_px: Some(batch.source.width_px),
            image_height_px: Some(batch.source.height_px),
            confidence: Some(detection.confidence),
        }
    } else {
        CueState {
            status: CueStatus::NotObserved,
            last_observation: Some(observation.clone()),
            image_center_px: None,
            image_space: Some(image_space),
            image_width_px: Some(batch.source.width_px),
            image_height_px: Some(batch.source.height_px),
            confidence: None,
        }
    }
}

fn parse_timestamp_ticks(timestamp: &BatchTimestamp) -> Result<u128, TrackingError> {
    timestamp
        .ticks
        .parse::<u128>()
        .map_err(|_| TrackingError::InvalidSourceTimestamp)
}

fn map_source_timestamp(timestamp: &BatchTimestamp) -> Result<Timestamp, TrackingError> {
    let stage = match timestamp.stage {
        BatchTimestampStage::Exposure => TimestampStage::Exposure,
        BatchTimestampStage::CameraReadComplete => TimestampStage::CameraReadComplete,
        BatchTimestampStage::SharedMemoryPublication => TimestampStage::SharedMemoryPublication,
        BatchTimestampStage::FrameArrival => TimestampStage::FrameArrival,
        BatchTimestampStage::InferenceStarted
        | BatchTimestampStage::InferenceComplete
        | BatchTimestampStage::BatchPublication => {
            return Err(TrackingError::InvalidSourceTimestamp);
        }
    };
    Ok(Timestamp {
        ticks: timestamp.ticks.clone(),
        unit: match timestamp.unit {
            BatchTimeUnit::Nanoseconds => TimeUnit::Nanoseconds,
        },
        clock: map_clock_kind(timestamp.clock),
        domain_id: timestamp.domain_id.clone(),
        stage,
    })
}

fn map_clock_kind(clock: BatchClockKind) -> crate::ClockKind {
    match clock {
        BatchClockKind::Monotonic => crate::ClockKind::Monotonic,
        BatchClockKind::UnixEpoch => crate::ClockKind::UnixEpoch,
        BatchClockKind::Sensor => crate::ClockKind::Sensor,
    }
}

fn map_image_space(image_space: BatchImageSpace) -> ImageSpace {
    match image_space {
        BatchImageSpace::Raw => ImageSpace::Raw,
        BatchImageSpace::Rectified => ImageSpace::Rectified,
    }
}

fn euclidean_distance(left: Point2d, right: Point2d) -> f64 {
    (left.x - right.x).hypot(left.y - right.y)
}

/// Rectangular Hungarian assignment. Each row is assigned to one of the real
/// detection columns or a per-row dummy column supplied by the caller.
fn minimum_cost_assignment(costs: &[Vec<f64>]) -> Vec<usize> {
    let row_count = costs.len();
    if row_count == 0 {
        return Vec::new();
    }
    let column_count = costs[0].len();
    let mut row_potential = vec![0.0_f64; row_count + 1];
    let mut column_potential = vec![0.0_f64; column_count + 1];
    let mut column_row = vec![0_usize; column_count + 1];
    let mut previous_column = vec![0_usize; column_count + 1];

    for row in 1..=row_count {
        column_row[0] = row;
        let mut current_column = 0;
        let mut minimum_reduced_cost = vec![f64::INFINITY; column_count + 1];
        let mut used_columns = vec![false; column_count + 1];
        loop {
            used_columns[current_column] = true;
            let current_row = column_row[current_column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0;
            for column in 1..=column_count {
                if used_columns[column] {
                    continue;
                }
                let reduced_cost = costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column];
                if reduced_cost < minimum_reduced_cost[column] {
                    minimum_reduced_cost[column] = reduced_cost;
                    previous_column[column] = current_column;
                }
                if minimum_reduced_cost[column] < delta {
                    delta = minimum_reduced_cost[column];
                    next_column = column;
                }
            }
            for column in 0..=column_count {
                if used_columns[column] {
                    row_potential[column_row[column]] += delta;
                    column_potential[column] -= delta;
                } else {
                    minimum_reduced_cost[column] -= delta;
                }
            }
            current_column = next_column;
            if column_row[current_column] == 0 {
                break;
            }
        }
        loop {
            let predecessor = previous_column[current_column];
            column_row[current_column] = column_row[predecessor];
            current_column = predecessor;
            if current_column == 0 {
                break;
            }
        }
    }

    let mut row_assignment = vec![column_count; row_count];
    for column in 1..=column_count {
        if column_row[column] > 0 {
            row_assignment[column_row[column] - 1] = column - 1;
        }
    }
    row_assignment
}

#[cfg(test)]
mod tests {
    use super::minimum_cost_assignment;

    #[test]
    fn minimum_cost_assignment_finds_global_one_to_one_solution() {
        let assignment =
            minimum_cost_assignment(&[vec![2.0, 2.0, 9.0, 9.0], vec![2.0, 6.0, 9.0, 9.0]]);
        assert_eq!(assignment, [1, 0]);
    }
}
