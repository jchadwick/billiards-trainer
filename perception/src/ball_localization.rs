//! Bounded, frame-local 2D ball localization on borrowed native RGB pixels.
//!
//! A caller supplies the intrinsics bound to the actual source geometry. Local
//! camera-ray rotation removes apparent off-axis stretching without estimating
//! ball depth, changing physical diameter, or modifying detector boxes. Every
//! proposal receives an explicit accepted/rejected result; fit diagnostics are
//! sensitivity measures, never calibrated confidence or physical uncertainty.

use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

use crate::ball_localization_profile::LoadedBallLocalizationProfile;

/// Guaranteed regulation-ball diameter used by every physical XY calibration.
pub const REGULATION_BALL_DIAMETER_MM: f64 = 57.15;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraIntrinsics {
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}
impl CameraIntrinsics {
    pub fn validate(&self) -> Result<(), BallLocalizationError> {
        if self.source_width_px < 2
            || self.source_height_px < 2
            || ![self.fx, self.fy, self.cx, self.cy]
                .iter()
                .all(|v| v.is_finite())
            || self.fx <= 0.0
            || self.fy <= 0.0
        {
            return Err(BallLocalizationError::InvalidIntrinsics);
        }
        Ok(())
    }
}

/// All image heuristics and work limits are configurable. Defaults are initial
/// engineering gates, not a statement that this achieves physical XY accuracy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BallLocalizationConfig {
    pub max_proposals: usize,
    pub reference_width_px: u32,
    pub patch_size_px: u32,
    pub minimum_supported_ray_fraction: f64,
    pub minimum_alternative_support_fraction: f64,
    /// Number of spatially coherent competing RANSAC samples required before
    /// treating a second plausible center as genuine boundary ambiguity.
    pub minimum_competing_fit_votes: usize,
    /// Alternative circle hypotheses must be within this factor of the best
    /// residual before they can make the boundary ambiguous.
    pub maximum_alternative_error_ratio: f64,
    pub minimum_background_supported_fraction: f64,
    pub batch_budget_ms: u64,
    pub proposal_budget_ms: u64,
    pub max_roi_side_px: u32,
    pub roi_padding_factor: f64,
    pub angular_samples: usize,
    pub radial_samples: usize,
    pub candidates_per_ray: usize,
    pub max_iterations: usize,
    pub ransac_iterations: usize,
    pub irls_iterations: usize,
    pub outline_samples: usize,
    pub radius_min_fraction: f64,
    pub radius_max_fraction: f64,
    pub minimum_radius_px: f64,
    pub maximum_radius_px: f64,
    pub gradient_step_px: f64,
    pub background_probe_px: f64,
    pub annulus_inner_factor: f64,
    pub annulus_outer_factor: f64,
    pub minimum_gradient: f64,
    pub minimum_foreground_background_contrast: f64,
    pub maximum_outer_background_distance: f64,
    pub inlier_threshold_px: f64,
    pub minimum_inliers: usize,
    pub angular_sectors: usize,
    pub minimum_coverage: f64,
    pub maximum_gap_fraction: f64,
    pub maximum_residual_px: f64,
    pub maximum_center_shift_fraction: f64,
    pub maximum_center_spread_px: f64,
    pub minimum_foreground_fraction: f64,
    pub irls_tuning: f64,
    pub irls_scale_floor_px: f64,
    pub convergence_px: f64,
    pub numerical_epsilon: f64,
    pub seed: u64,
}
impl Default for BallLocalizationConfig {
    fn default() -> Self {
        Self {
            max_proposals: 16,
            reference_width_px: 3840,
            patch_size_px: 193,
            minimum_supported_ray_fraction: 0.5,
            minimum_alternative_support_fraction: 0.95,
            minimum_competing_fit_votes: 3,
            maximum_alternative_error_ratio: 1.1,
            minimum_background_supported_fraction: 0.6,
            batch_budget_ms: 25,
            proposal_budget_ms: 10,
            max_roi_side_px: 512,
            roi_padding_factor: 1.8,
            angular_samples: 96,
            radial_samples: 48,
            candidates_per_ray: 3,
            max_iterations: 3,
            ransac_iterations: 128,
            irls_iterations: 10,
            outline_samples: 48,
            radius_min_fraction: 0.6,
            radius_max_fraction: 1.4,
            minimum_radius_px: 4.0,
            maximum_radius_px: 160.0,
            gradient_step_px: 1.0,
            background_probe_px: 3.0,
            annulus_inner_factor: 1.5,
            annulus_outer_factor: 1.8,
            minimum_gradient: 0.035,
            minimum_foreground_background_contrast: 0.06,
            maximum_outer_background_distance: 0.25,
            inlier_threshold_px: 1.5,
            minimum_inliers: 28,
            angular_sectors: 16,
            minimum_coverage: 2.0 / 3.0,
            maximum_gap_fraction: 1.0 / 3.0,
            maximum_residual_px: 1.5,
            maximum_center_shift_fraction: 0.2,
            maximum_center_spread_px: 1.5,
            minimum_foreground_fraction: 0.4,
            irls_tuning: 2.5,
            irls_scale_floor_px: 0.25,
            convergence_px: 0.01,
            numerical_epsilon: 1e-9,
            seed: 716,
        }
    }
}
// Absolute safety ceilings are structural, not tuning defaults. They bound
// allocations and loops even when configuration comes from untrusted JSON.
const MAX_PROPOSALS: usize = 256;
const MAX_PATCH_SIDE: u32 = 513;
const MAX_ROI_SIDE: u32 = 2048;
const MAX_ANGULAR_SAMPLES: usize = 512;
const MAX_RADIAL_SAMPLES: usize = 512;
const MAX_CANDIDATES_PER_RAY: usize = 8;
const MAX_RANSAC_ITERATIONS: usize = 1024;
const MAX_IRLS_ITERATIONS: usize = 64;
const MAX_OUTLINE_SAMPLES: usize = 512;
const MAX_ANGULAR_SECTORS: usize = 128;
const MAX_BUDGET_MS: u64 = 10_000;

impl BallLocalizationConfig {
    pub fn validate(&self) -> Result<(), BallLocalizationError> {
        let positive = [
            self.roi_padding_factor,
            self.radius_min_fraction,
            self.radius_max_fraction,
            self.minimum_radius_px,
            self.maximum_radius_px,
            self.gradient_step_px,
            self.background_probe_px,
            self.annulus_inner_factor,
            self.annulus_outer_factor,
            self.minimum_gradient,
            self.minimum_foreground_background_contrast,
            self.maximum_outer_background_distance,
            self.inlier_threshold_px,
            self.maximum_residual_px,
            self.maximum_center_shift_fraction,
            self.maximum_center_spread_px,
            self.irls_tuning,
            self.irls_scale_floor_px,
            self.convergence_px,
            self.numerical_epsilon,
        ];
        let fractions = [
            self.minimum_alternative_support_fraction,
            self.minimum_supported_ray_fraction,
            self.minimum_background_supported_fraction,
            self.minimum_coverage,
            self.maximum_gap_fraction,
            self.minimum_foreground_fraction,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || fractions
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0 || *v > 1.0)
            || self.max_proposals > MAX_PROPOSALS
            || self.patch_size_px > MAX_PATCH_SIDE
            || self.patch_size_px.is_multiple_of(2)
            || self.max_roi_side_px > MAX_ROI_SIDE
            || self.angular_samples > MAX_ANGULAR_SAMPLES
            || self.radial_samples > MAX_RADIAL_SAMPLES
            || self.candidates_per_ray > MAX_CANDIDATES_PER_RAY
            || self.ransac_iterations > MAX_RANSAC_ITERATIONS
            || self.minimum_competing_fit_votes == 0
            || self.minimum_competing_fit_votes > self.ransac_iterations
            || self.irls_iterations > MAX_IRLS_ITERATIONS
            || self.outline_samples > MAX_OUTLINE_SAMPLES
            || self.angular_sectors > MAX_ANGULAR_SECTORS
            || self.batch_budget_ms > MAX_BUDGET_MS
            || self.proposal_budget_ms > MAX_BUDGET_MS
            || self.max_proposals == 0
            || self.reference_width_px == 0
            || self.patch_size_px < 3
            || self.max_roi_side_px < 2
            || self.roi_padding_factor < 1.0
            || self.angular_samples < self.minimum_inliers
            || self.minimum_inliers < 3
            || self.angular_sectors < 3
            || self.angular_sectors > self.angular_samples
            || self.radial_samples < 3
            || self.candidates_per_ray == 0
            || self.candidates_per_ray > self.radial_samples
            || self.max_iterations == 0
            || self.max_iterations > 3
            || self.ransac_iterations == 0
            || self.irls_iterations == 0
            || self.outline_samples < 4
            || self.radius_min_fraction >= self.radius_max_fraction
            || self.minimum_radius_px >= self.maximum_radius_px
            || self.annulus_inner_factor <= self.radius_max_fraction
            || self.annulus_outer_factor <= self.annulus_inner_factor
            || self.background_probe_px < self.gradient_step_px
            || self
                .angular_samples
                .checked_mul(self.candidates_per_ray)
                .is_none()
            || self.numerical_epsilon >= self.convergence_px
            || !self.maximum_alternative_error_ratio.is_finite()
            || self.maximum_alternative_error_ratio < 1.0
        {
            return Err(BallLocalizationError::InvalidConfiguration);
        }
        // Duration conversion is checked here rather than panicking at runtime.
        if Instant::now()
            .checked_add(Duration::from_millis(self.batch_budget_ms))
            .is_none()
            || Instant::now()
                .checked_add(Duration::from_millis(self.proposal_budget_ms))
                .is_none()
        {
            return Err(BallLocalizationError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BallProposal {
    pub detection_index: u64,
    pub bbox: [f64; 4],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementStatus {
    Accepted,
    Rejected,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionReason {
    InvalidProposal,
    ProposalLimitExceeded,
    RoiTooLarge,
    InsufficientBoundary,
    AmbiguousBoundary,
    InadequateCoverage,
    ExcessiveResidual,
    ExcessiveCenterShift,
    UnstableCenter,
    InsufficientForeground,
    DeadlineExceeded,
    NumericalFailure,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BallLocalizationQuality {
    /// Last reached or failing stage in the current iteration. Measurements
    /// from earlier stages in that iteration can remain populated; `None`
    /// means the corresponding measurement was not computed in this iteration.
    pub stage: BallFitStage,
    pub fit_id: Option<usize>,
    pub boundary_candidates: Option<usize>,
    pub supported_ray_fraction: Option<f64>,
    pub background_sample_fraction: Option<f64>,
    pub background_supported_fraction: Option<f64>,
    pub inliers: Option<usize>,
    pub angular_coverage: Option<f64>,
    pub largest_gap_fraction: Option<f64>,
    pub residual_rms_px: Option<f64>,
    pub center_shift_px: Option<f64>,
    pub center_spread_px: Option<f64>,
    pub foreground_fraction: Option<f64>,
    pub iterations: usize,
}

impl BallLocalizationQuality {
    fn begin_iteration(&mut self, iteration: usize) {
        // Reset before the deadline check: a timeout at the next iteration
        // must never inherit the previous iteration's successful measurements.
        *self = Self {
            stage: BallFitStage::IterationStart,
            iterations: iteration,
            ..Self::default()
        };
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BallFitStage {
    #[default]
    ProposalValidation,
    ProposalCapacity,
    CropSetup,
    IterationStart,
    ViewGeometry,
    BackgroundSampling,
    BoundarySampling,
    HypothesisGeneration,
    AmbiguityAssessment,
    FitRefinement,
    Coverage,
    Residual,
    CenterProjection,
    CenterStability,
    Foreground,
    OutlineProjection,
    Complete,
}

/// Audit events are a bounded research record of the same estimator used by
/// live and replay. Local coordinates belong to the current virtual patch;
/// projected points use native source-image pixels and may be unavailable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BallLocalizationAuditEvent {
    pub detection_index: u64,
    pub iteration: Option<usize>,
    pub stage: BallFitStage,
    pub data: BallLocalizationAuditEventData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BallLocalizationAuditEventData {
    IterationStarted {
        axis_image_px: [f64; 2],
    },
    BoundaryCandidate {
        ray: usize,
        point_local_px: [f64; 2],
        point_image_px: Option<[f64; 2]>,
        strength: f64,
    },
    RawHypothesis {
        fit_id: usize,
        center_local_px: [f64; 2],
        center_image_px: Option<[f64; 2]>,
        radius_local_px: f64,
        support: usize,
        squared_error_local_px: f64,
    },
    SelectedInlier {
        fit_id: usize,
        ray: usize,
        point_local_px: [f64; 2],
        point_image_px: Option<[f64; 2]>,
    },
    RefinedFit {
        fit_id: usize,
        center_local_px: [f64; 2],
        center_image_px: Option<[f64; 2]>,
        radius_local_px: f64,
        outline_image_px: Vec<Option<[f64; 2]>>,
    },
    Outcome {
        status: RefinementStatus,
        reason: Option<RejectionReason>,
        quality: Box<BallLocalizationQuality>,
        center_image_px: Option<[f64; 2]>,
        outline: Option<Box<BallOutline>>,
    },
}

/// The cap is a structural safety limit, not a production estimator setting.
/// Payload sizes are further bounded by the validated sample/outline limits.
pub const MAX_BALL_LOCALIZATION_AUDIT_EVENTS: usize = 8192;
/// Trace storage is also bounded by bytes, independently of estimator/sample
/// limits. This covers reserved event slots and owned variable-length arrays.
pub const MAX_BALL_LOCALIZATION_AUDIT_BYTES: usize = 8 * 1024 * 1024;
pub const BALL_LOCALIZATION_AUDIT_VERSION: u32 = 1;

/// Per-audit limits control diagnostics only; they do not select a production
/// algorithm or change estimator configuration. Both are capped independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallLocalizationAuditLimits {
    pub maximum_events: usize,
    pub maximum_bytes: usize,
}

impl Default for BallLocalizationAuditLimits {
    fn default() -> Self {
        Self {
            maximum_events: MAX_BALL_LOCALIZATION_AUDIT_EVENTS,
            maximum_bytes: MAX_BALL_LOCALIZATION_AUDIT_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BallLocalizationAuditTrace {
    pub schema_version: u32,
    pub maximum_events: usize,
    pub maximum_bytes: usize,
    pub retained_bytes: usize,
    pub dropped_events: usize,
    pub events: Vec<BallLocalizationAuditEvent>,
}

impl BallLocalizationAuditTrace {
    fn record(&mut self, event: impl FnOnce() -> BallLocalizationAuditEvent) {
        if self.events.len() >= self.maximum_events {
            self.dropped_events = self.dropped_events.saturating_add(1);
            return;
        }
        let event = event();
        let extra = match &event.data {
            BallLocalizationAuditEventData::RefinedFit {
                outline_image_px, ..
            } => outline_image_px.capacity() * std::mem::size_of::<Option<[f64; 2]>>(),
            BallLocalizationAuditEventData::Outcome {
                outline: Some(outline),
                ..
            } => {
                std::mem::size_of::<BallLocalizationQuality>()
                    + std::mem::size_of::<BallOutline>()
                    + outline.points_image_px.capacity() * std::mem::size_of::<[f64; 2]>()
            }
            BallLocalizationAuditEventData::IterationStarted { .. }
            | BallLocalizationAuditEventData::BoundaryCandidate { .. }
            | BallLocalizationAuditEventData::RawHypothesis { .. }
            | BallLocalizationAuditEventData::SelectedInlier { .. }
            | BallLocalizationAuditEventData::Outcome { outline: None, .. } => {
                std::mem::size_of::<BallLocalizationQuality>()
            }
        };
        let Some(retained_bytes) = self.retained_bytes.checked_add(extra) else {
            self.dropped_events = self.dropped_events.saturating_add(1);
            return;
        };
        if retained_bytes > self.maximum_bytes {
            self.dropped_events = self.dropped_events.saturating_add(1);
            return;
        }
        self.retained_bytes = retained_bytes;
        self.events.push(event);
    }
}
/// Projected fitted circle, in original image pixels. These are appearance
/// measurements only; they must never overwrite the fixed physical radius.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BallOutline {
    pub points_image_px: Vec<[f64; 2]>,
    pub half_extent_image_px: [f64; 2],
    pub local_radius_px: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BallRefinement {
    pub detection_index: u64,
    pub center_image_px: Option<[f64; 2]>,
    pub status: RefinementStatus,
    pub reason: Option<RejectionReason>,
    pub quality: BallLocalizationQuality,
    pub outline: Option<BallOutline>,
}

/// Source/session identities assigned by the single-flight worker after the
/// detector returns. Keeping this alongside the crop results prevents a valid
/// fit from being silently reused for a neighboring frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallRefinementIdentity {
    pub source_session_id: String,
    pub frame_id: String,
    pub configuration_revision: u64,
    pub calibration_revision: u64,
    pub motion_generation: u64,
}

/// Refinements travel with the exact inference output and owned source frame.
/// Dimensions and profile revisions are checked again before tracking accepts
/// any center.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BallRefinementBatch {
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub camera_profile_id: String,
    pub image_space: crate::detection_batch::ImageSpace,
    pub configuration_revision: u64,
    pub optics_profile_id: String,
    pub optics_revision: u64,
    pub refiner_revision: u64,
    pub table_revision: u64,
    pub refinement_elapsed_ms: f64,
    pub identity: Option<BallRefinementIdentity>,
    pub observations: Vec<BallRefinement>,
}

/// The one image center selected for tracking, with a physical table position
/// only when the same accepted crop fit passes the matching center calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedBallMeasurement {
    pub detection_index: u64,
    pub center_image_px: [f64; 2],
    pub table_center_mm: Option<[f64; 2]>,
    pub refinement_status: Option<RefinementStatus>,
}

/// Frame-bound selected measurements consumed together by association and
/// coordinate conversion. The original DetectionBatchV1 is kept unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedBallMeasurementBatch {
    pub engine_instance_id: String,
    pub source_session_id: String,
    pub frame_id: String,
    pub batch_sequence: u64,
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub image_space: crate::detection_batch::ImageSpace,
    pub camera_profile_id: String,
    pub configuration_revision: u64,
    pub rectification_revision: u64,
    pub optics_profile_id: Option<String>,
    pub optics_revision: Option<u64>,
    pub refiner_revision: Option<u64>,
    pub table_revision: u64,
    pub calibration_revision: Option<u64>,
    pub measurements: Vec<ResolvedBallMeasurement>,
}

/// Inputs used to validate one refinement sidecar and map its selected centers.
/// Keeping these values together prevents live and replay callers from
/// accidentally omitting one part of the frame/profile identity.
pub struct BallMeasurementResolutionContext<'a> {
    pub ball_class_id: u32,
    pub camera_profile_id: &'a str,
    pub configuration_revision: u64,
    pub rectification_revision: u64,
    pub profile: Option<&'a LoadedBallLocalizationProfile>,
    pub refinements: Option<&'a BallRefinementBatch>,
    pub expected_identity: Option<&'a BallRefinementIdentity>,
    pub require_identity: bool,
}

impl ResolvedBallMeasurementBatch {
    /// Build a frame-bound coarse sidecar for consumers that do not have a
    /// validated native-pixel refinement profile. The original detection
    /// midpoint remains the visible image location, while physical XY stays
    /// unavailable. This keeps replay and live tracking on the same validated
    /// measurement contract without claiming that a box midpoint is a fitted
    /// ball center.
    pub fn from_detection_batch_midpoints(
        batch: &crate::detection_batch::DetectionBatchV1,
        ball_class_id: u32,
        camera_profile_id: impl Into<String>,
        configuration_revision: u64,
        rectification_revision: u64,
    ) -> Self {
        let measurements = batch
            .detections
            .iter()
            .filter(|detection| detection.class_id == ball_class_id)
            .map(|detection| ResolvedBallMeasurement {
                detection_index: detection.detection_index,
                center_image_px: [detection.center_image_px.x, detection.center_image_px.y],
                table_center_mm: None,
                refinement_status: None,
            })
            .collect();
        Self {
            engine_instance_id: batch.engine_instance_id.clone(),
            source_session_id: batch.source.session_id.clone(),
            frame_id: batch.source.frame_id.clone(),
            batch_sequence: batch.batch_sequence,
            source_width_px: batch.source.width_px,
            source_height_px: batch.source.height_px,
            image_space: batch.image_space,
            camera_profile_id: camera_profile_id.into(),
            configuration_revision,
            rectification_revision,
            optics_profile_id: None,
            optics_revision: None,
            refiner_revision: None,
            table_revision: 0,
            calibration_revision: None,
            measurements,
        }
    }
}

/// Validate refinement provenance, select each ball center once, and apply
/// only its matching ball-center calibration. Invalid or absent refinement
/// data degrades to the raw detector midpoint for visibility; it never falls
/// through to a cloth-corner transform.
pub fn resolve_ball_measurements_for_batch(
    batch: &crate::detection_batch::DetectionBatchV1,
    context: BallMeasurementResolutionContext<'_>,
) -> ResolvedBallMeasurementBatch {
    let BallMeasurementResolutionContext {
        ball_class_id,
        camera_profile_id,
        configuration_revision,
        rectification_revision,
        profile,
        refinements,
        expected_identity,
        require_identity,
    } = context;
    let balls = batch
        .detections
        .iter()
        .filter(|detection| detection.class_id == ball_class_id)
        .collect::<Vec<_>>();
    let optics = profile.and_then(|profile| profile.optics.as_ref());
    let identity_matches = refinements.is_some_and(|refinements| {
        match (refinements.identity.as_ref(), expected_identity) {
            (Some(actual), Some(expected)) => {
                actual.source_session_id == expected.source_session_id
                    && actual.frame_id == expected.frame_id
                    && actual.configuration_revision == expected.configuration_revision
                    && actual.calibration_revision == expected.calibration_revision
            }
            (None, Some(_)) | (None, None) => !require_identity,
            _ => false,
        }
    });
    let profile_matches = refinements.is_some_and(|refinements| {
        optics.is_some_and(|optics| {
            refinements.camera_profile_id == camera_profile_id
                && refinements.image_space == batch.image_space
                && refinements.configuration_revision == configuration_revision
                && refinements.source_width_px == batch.source.width_px
                && refinements.source_height_px == batch.source.height_px
                && refinements.optics_profile_id == optics.optics_profile_id
                && refinements.optics_revision == optics.optics_revision
                && refinements.refiner_revision == optics.refiner_revision
                && refinements.table_revision
                    == profile.map_or(0, |profile| profile.settings.table_revision)
        })
    });
    let refinement_list_matches = refinements.is_some_and(|refinements| {
        refinements.observations.len() == balls.len()
            && refinements
                .observations
                .iter()
                .zip(&balls)
                .all(|(observation, detection)| {
                    observation.detection_index == detection.detection_index
                        && match observation.status {
                            RefinementStatus::Accepted => {
                                observation.center_image_px.is_some_and(|point| {
                                    point.iter().all(|value| value.is_finite())
                                        && point[0] >= 0.0
                                        && point[1] >= 0.0
                                        && point[0] < f64::from(batch.source.width_px)
                                        && point[1] < f64::from(batch.source.height_px)
                                })
                            }
                            RefinementStatus::Rejected => observation.center_image_px.is_none(),
                        }
                })
    });
    let accept_refinement = identity_matches && profile_matches && refinement_list_matches;

    let measurements = balls
        .iter()
        .enumerate()
        .map(|(index, detection)| {
            let refinement = accept_refinement
                .then(|| refinements.map(|batch| &batch.observations[index]))
                .flatten();
            let center = refinement
                .and_then(|refinement| {
                    (refinement.status == RefinementStatus::Accepted)
                        .then_some(refinement.center_image_px)
                        .flatten()
                })
                .unwrap_or([detection.center_image_px.x, detection.center_image_px.y]);
            let table_center_mm = refinement
                .filter(|refinement| refinement.status == RefinementStatus::Accepted)
                .and_then(|_| {
                    let calibration = profile?.calibration.as_ref()?;
                    let context = profile?.calibration_context(camera_profile_id)?;
                    calibration
                        .project(
                            crate::calibration::ImagePointPx {
                                x_px: center[0],
                                y_px: center[1],
                            },
                            context,
                        )
                        .ok()
                        .map(|table| [table.x_mm, table.y_mm])
                });
            ResolvedBallMeasurement {
                detection_index: detection.detection_index,
                center_image_px: center,
                table_center_mm,
                refinement_status: refinement.map(|refinement| refinement.status),
            }
        })
        .collect();

    ResolvedBallMeasurementBatch {
        engine_instance_id: batch.engine_instance_id.clone(),
        source_session_id: batch.source.session_id.clone(),
        frame_id: batch.source.frame_id.clone(),
        batch_sequence: batch.batch_sequence,
        source_width_px: batch.source.width_px,
        source_height_px: batch.source.height_px,
        image_space: batch.image_space,
        camera_profile_id: camera_profile_id.to_owned(),
        configuration_revision,
        rectification_revision,
        optics_profile_id: optics.map(|optics| optics.optics_profile_id.clone()),
        optics_revision: optics.map(|optics| optics.optics_revision),
        refiner_revision: optics.map(|optics| optics.refiner_revision),
        table_revision: profile.map_or(0, |profile| profile.settings.table_revision),
        calibration_revision: profile
            .and_then(|profile| profile.calibration.as_ref())
            .map(|calibration| calibration.as_asset().calibration_revision),
        measurements,
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BallLocalizationError {
    InvalidIntrinsics,
    InvalidConfiguration,
    InvalidRgbBuffer,
    TooManyProposals,
    DuplicateDetectionIndex,
    InvalidAuditLimits,
}
impl Display for BallLocalizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ball localization: {self:?}")
    }
}
impl std::error::Error for BallLocalizationError {}

#[derive(Clone, Copy)]
struct View {
    r: [[f64; 3]; 3],
    focal: [f64; 2],
}
impl View {
    fn new(k: &CameraIntrinsics, center: [f64; 2]) -> Self {
        let mut d = [(center[0] - k.cx) / k.fx, (center[1] - k.cy) / k.fy, 1.0];
        let n = (d.iter().map(|v| v * v).sum::<f64>()).sqrt();
        for v in &mut d {
            *v /= n;
        }
        // Rodrigues(d -> optical axis), with v=d cross [0,0,1]. No depth fit.
        let v = [d[1], -d[0], 0.0];
        let s = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = f64::from(i == j) + s[i][j];
            }
            for (q, s_row) in s.iter().enumerate() {
                let scale = s[i][q] / (1.0 + d[2]);
                for (j, value) in row.iter_mut().enumerate() {
                    *value += scale * s_row[j];
                }
            }
        }
        Self {
            r,
            focal: [k.fx, k.fy],
        }
    }
    fn local(&self, k: &CameraIntrinsics, p: [f64; 2]) -> Option<[f64; 2]> {
        let d = [(p[0] - k.cx) / k.fx, (p[1] - k.cy) / k.fy, 1.0];
        let mut q = [0.0; 3];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (0..3).map(|j| self.r[i][j] * d[j]).sum();
        }
        if q[2] <= 0.0 {
            return None;
        }
        Some([self.focal[0] * q[0] / q[2], self.focal[1] * q[1] / q[2]])
    }
    fn source(&self, k: &CameraIntrinsics, p: [f64; 2]) -> Option<[f64; 2]> {
        let d = [p[0] / self.focal[0], p[1] / self.focal[1], 1.0];
        let mut q = [0.0; 3];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (0..3).map(|j| self.r[j][i] * d[j]).sum();
        }
        if q[2] <= 0.0 {
            return None;
        }
        let p = [k.fx * q[0] / q[2] + k.cx, k.fy * q[1] / q[2] + k.cy];
        p.iter().all(|v| v.is_finite()).then_some(p)
    }
}
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn color_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>() / 3.0).sqrt()
}
fn bilinear(rgb: &[u8], k: &CameraIntrinsics, p: [f64; 2], roi: [f64; 4]) -> Option<[f64; 3]> {
    let [x, y] = p;
    if x < roi[0]
        || y < roi[1]
        || x > roi[2]
        || y > roi[3]
        || x < 0.0
        || y < 0.0
        || x >= f64::from(k.source_width_px - 1)
        || y >= f64::from(k.source_height_px - 1)
    {
        return None;
    }
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    let tx = x - ix as f64;
    let ty = y - iy as f64;
    let w = k.source_width_px as usize;
    let mut out = [0.0; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let p00 = f64::from(rgb[(iy * w + ix) * 3 + channel]);
        let p10 = f64::from(rgb[(iy * w + ix + 1) * 3 + channel]);
        let p01 = f64::from(rgb[((iy + 1) * w + ix) * 3 + channel]);
        let p11 = f64::from(rgb[((iy + 1) * w + ix + 1) * 3 + channel]);
        *value = ((1.0 - ty) * (p00 * (1.0 - tx) + p10 * tx) + ty * (p01 * (1.0 - tx) + p11 * tx))
            / f64::from(u8::MAX);
    }
    Some(out)
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}
fn deadline(now: Instant, end: Instant) -> Result<(), RejectionReason> {
    if now >= end {
        Err(RejectionReason::DeadlineExceeded)
    } else {
        Ok(())
    }
}

fn start_iteration(
    quality: &mut BallLocalizationQuality,
    iteration: usize,
    now: Instant,
    end: Instant,
) -> Result<(), RejectionReason> {
    quality.begin_iteration(iteration);
    deadline(now, end)
}

#[derive(Clone, Copy)]
struct Candidate {
    point: [f64; 2],
    ray: usize,
    strength: f64,
}
#[derive(Clone, Copy, Debug)]
struct Circle {
    center: [f64; 2],
    radius: f64,
}
fn three_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], eps: f64) -> Option<Circle> {
    let b = [b[0] - a[0], b[1] - a[1]];
    let c = [c[0] - a[0], c[1] - a[1]];
    let det = 2.0 * (b[0] * c[1] - b[1] * c[0]);
    if det.abs() <= eps {
        return None;
    }
    let bn = b[0] * b[0] + b[1] * b[1];
    let cn = c[0] * c[0] + c[1] * c[1];
    let d = [(bn * c[1] - cn * b[1]) / det, (b[0] * cn - c[0] * bn) / det];
    let circle = Circle {
        center: [a[0] + d[0], a[1] + d[1]],
        radius: d[0].hypot(d[1]),
    };
    circle.radius.is_finite().then_some(circle)
}
fn solve(mut m: [[f64; 3]; 3], mut y: [f64; 3], eps: f64) -> Option<[f64; 3]> {
    for i in 0..3 {
        let pivot = (i..3).max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs()))?;
        if m[pivot][i].abs() <= eps {
            return None;
        }
        m.swap(i, pivot);
        y.swap(i, pivot);
        let value = m[i][i];
        for entry in m[i].iter_mut().skip(i) {
            *entry /= value;
        }
        y[i] /= value;
        let pivot_row = m[i];
        for q in 0..3 {
            if q != i {
                let f = m[q][i];
                for (j, entry) in m[q].iter_mut().enumerate().skip(i) {
                    *entry -= f * pivot_row[j];
                }
                y[q] -= f * y[i];
            }
        }
    }
    y.iter().all(|v| v.is_finite()).then_some(y)
}
fn irls(
    mut fit: Circle,
    points: &[[f64; 2]],
    cfg: &BallLocalizationConfig,
    end: Instant,
) -> Result<Circle, RejectionReason> {
    for _ in 0..cfg.irls_iterations {
        deadline(Instant::now(), end)?;
        let residuals: Vec<f64> = points
            .iter()
            .map(|&p| distance(p, fit.center) - fit.radius)
            .collect();
        let mut absolute: Vec<f64> = residuals.iter().map(|v| v.abs()).collect();
        let scale = median(&mut absolute).max(cfg.irls_scale_floor_px);
        let mut matrix = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for (&p, residual) in points.iter().zip(residuals) {
            let d = distance(p, fit.center).max(cfg.numerical_epsilon);
            let row = [(fit.center[0] - p[0]) / d, (fit.center[1] - p[1]) / d, -1.0];
            let weight = 1.0 / (1.0 + (residual / (cfg.irls_tuning * scale)).powi(2));
            for i in 0..3 {
                rhs[i] -= weight * row[i] * residual;
                for j in 0..3 {
                    matrix[i][j] += weight * row[i] * row[j];
                }
            }
        }
        let step =
            solve(matrix, rhs, cfg.numerical_epsilon).ok_or(RejectionReason::NumericalFailure)?;
        fit.center[0] += step[0];
        fit.center[1] += step[1];
        fit.radius += step[2];
        if fit.radius <= 0.0 || !fit.radius.is_finite() {
            return Err(RejectionReason::NumericalFailure);
        }
        if step.iter().map(|v| v * v).sum::<f64>().sqrt() < cfg.convergence_px {
            break;
        }
    }
    Ok(fit)
}
fn select(candidates: &[Candidate], fit: Circle, cfg: &BallLocalizationConfig) -> Vec<Candidate> {
    let mut chosen = vec![None::<(f64, Candidate)>; cfg.angular_samples];
    for &c in candidates {
        let residual = (distance(c.point, fit.center) - fit.radius).abs();
        if residual <= cfg.inlier_threshold_px && chosen[c.ray].is_none_or(|(r, _)| residual < r) {
            chosen[c.ray] = Some((residual, c));
        }
    }
    chosen.into_iter().flatten().map(|(_, c)| c).collect()
}

fn has_competing_fit_cluster(
    hypotheses: &[(Circle, usize, f64)],
    best: Circle,
    best_support: usize,
    best_error: f64,
    config: &BallLocalizationConfig,
    end: Instant,
) -> Result<bool, RejectionReason> {
    let error_limit = best_error * config.maximum_alternative_error_ratio
        + config.numerical_epsilon * best_support as f64;
    let plausible = hypotheses
        .iter()
        .filter(|(_, support, error)| {
            *support as f64 >= best_support as f64 * config.minimum_alternative_support_fraction
                && *error <= error_limit
        })
        .collect::<Vec<_>>();

    for (candidate, _, _) in &plausible {
        deadline(Instant::now(), end)?;
        if distance(candidate.center, best.center) <= config.maximum_center_spread_px {
            continue;
        }
        let mut cluster_votes = 0;
        for (other, _, _) in &plausible {
            if distance(other.center, best.center) > config.maximum_center_spread_px
                && distance(other.center, candidate.center) <= config.maximum_center_spread_px
            {
                cluster_votes += 1;
                if cluster_votes >= config.minimum_competing_fit_votes {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

fn coverage(points: &[Candidate], center: [f64; 2], cfg: &BallLocalizationConfig) -> (f64, f64) {
    let mut occupied = vec![false; cfg.angular_sectors];
    let mut angles = Vec::new();
    for c in points {
        let angle = (c.point[1] - center[1])
            .atan2(c.point[0] - center[0])
            .rem_euclid(std::f64::consts::TAU);
        let sector = ((angle / std::f64::consts::TAU * cfg.angular_sectors as f64) as usize)
            .min(cfg.angular_sectors - 1);
        occupied[sector] = true;
        angles.push(angle);
    }
    angles.sort_by(f64::total_cmp);
    let mut largest = std::f64::consts::TAU;
    if !angles.is_empty() {
        largest = angles[0] + std::f64::consts::TAU - angles[angles.len() - 1];
        for pair in angles.windows(2) {
            largest = largest.max(pair[1] - pair[0]);
        }
    }
    (
        occupied.iter().filter(|v| **v).count() as f64 / cfg.angular_sectors as f64,
        largest / std::f64::consts::TAU,
    )
}
fn next_random(state: &mut u64, n: usize) -> usize {
    // Numerical Recipes LCG: algorithm constants, not camera/config parameters.
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 32) as usize) % n
}

/// Refine a complete proposal list. Batch-level input errors are explicit;
/// ordinary fit/coverage/budget failures return one rejected observation each.
pub fn refine_ball_batch(
    rgb: &[u8],
    intrinsics: &CameraIntrinsics,
    proposals: &[BallProposal],
    config: &BallLocalizationConfig,
) -> Result<Vec<BallRefinement>, BallLocalizationError> {
    refine_ball_batch_impl(rgb, intrinsics, proposals, config, None)
}

/// Run the production estimator while retaining a bounded diagnostic trace.
/// This audit entry point does not select a different fitting algorithm.
/// Trace generation takes wall time and can therefore consume a real deadline;
/// callers comparing geometry should supply sufficient diagnostic budgets.
pub fn refine_ball_batch_with_audit(
    rgb: &[u8],
    intrinsics: &CameraIntrinsics,
    proposals: &[BallProposal],
    config: &BallLocalizationConfig,
    limits: BallLocalizationAuditLimits,
) -> Result<(Vec<BallRefinement>, BallLocalizationAuditTrace), BallLocalizationError> {
    let Some(reserved_bytes) = limits
        .maximum_events
        .checked_mul(std::mem::size_of::<BallLocalizationAuditEvent>())
    else {
        return Err(BallLocalizationError::InvalidAuditLimits);
    };
    if limits.maximum_events == 0
        || limits.maximum_events > MAX_BALL_LOCALIZATION_AUDIT_EVENTS
        || limits.maximum_bytes == 0
        || limits.maximum_bytes > MAX_BALL_LOCALIZATION_AUDIT_BYTES
        || reserved_bytes > limits.maximum_bytes
    {
        return Err(BallLocalizationError::InvalidAuditLimits);
    }
    let events = Vec::<BallLocalizationAuditEvent>::with_capacity(limits.maximum_events);
    let Some(reserved_bytes) = events
        .capacity()
        .checked_mul(std::mem::size_of::<BallLocalizationAuditEvent>())
    else {
        return Err(BallLocalizationError::InvalidAuditLimits);
    };
    if reserved_bytes > limits.maximum_bytes {
        return Err(BallLocalizationError::InvalidAuditLimits);
    }
    let mut trace = BallLocalizationAuditTrace {
        schema_version: BALL_LOCALIZATION_AUDIT_VERSION,
        maximum_events: limits.maximum_events,
        maximum_bytes: limits.maximum_bytes,
        retained_bytes: reserved_bytes,
        dropped_events: 0,
        // Account for the actual reserved capacity so Vec's growth strategy
        // cannot add uncharged retained event storage later.
        events,
    };
    let results = refine_ball_batch_impl(rgb, intrinsics, proposals, config, Some(&mut trace))?;
    Ok((results, trace))
}

fn record_trace(
    trace: &mut Option<&mut BallLocalizationAuditTrace>,
    detection_index: u64,
    quality: &BallLocalizationQuality,
    data: impl FnOnce() -> BallLocalizationAuditEventData,
) {
    if let Some(trace) = trace.as_deref_mut() {
        trace.record(|| BallLocalizationAuditEvent {
            detection_index,
            iteration: (quality.iterations != 0).then_some(quality.iterations),
            stage: quality.stage,
            data: data(),
        });
    }
}

fn refine_ball_batch_impl(
    rgb: &[u8],
    intrinsics: &CameraIntrinsics,
    proposals: &[BallProposal],
    config: &BallLocalizationConfig,
    mut trace: Option<&mut BallLocalizationAuditTrace>,
) -> Result<Vec<BallRefinement>, BallLocalizationError> {
    intrinsics.validate()?;
    config.validate()?;
    let expected = (intrinsics.source_width_px as usize)
        .checked_mul(intrinsics.source_height_px as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or(BallLocalizationError::InvalidRgbBuffer)?;
    if rgb.len() != expected {
        return Err(BallLocalizationError::InvalidRgbBuffer);
    }
    if proposals.len() > MAX_PROPOSALS {
        return Err(BallLocalizationError::TooManyProposals);
    }
    for (i, p) in proposals.iter().enumerate() {
        if proposals[..i]
            .iter()
            .any(|q| p.detection_index == q.detection_index)
        {
            return Err(BallLocalizationError::DuplicateDetectionIndex);
        }
    }
    // Pixel gates are expressed at a declared reference source width. Ratios,
    // angular counts and runtime budgets are independent of resolution.
    let scale = f64::from(intrinsics.source_width_px) / f64::from(config.reference_width_px);
    let mut scaled = config.clone();
    scaled.max_roi_side_px = (f64::from(config.max_roi_side_px) * scale).ceil().max(2.0) as u32;
    scaled.patch_size_px = (f64::from(config.patch_size_px) * scale).ceil().max(3.0) as u32;
    if scaled.max_roi_side_px > MAX_ROI_SIDE || scaled.patch_size_px > MAX_PATCH_SIDE {
        return Err(BallLocalizationError::InvalidConfiguration);
    }
    scaled.minimum_radius_px *= scale;
    scaled.maximum_radius_px *= scale;
    scaled.gradient_step_px *= scale;
    scaled.background_probe_px *= scale;
    scaled.inlier_threshold_px *= scale;
    scaled.maximum_residual_px *= scale;
    scaled.maximum_center_spread_px *= scale;
    scaled.irls_scale_floor_px *= scale;
    scaled.convergence_px *= scale;
    let batch_end = Instant::now() + Duration::from_millis(config.batch_budget_ms);
    let mut output = Vec::with_capacity(proposals.len());
    for proposal in proposals.iter().take(config.max_proposals) {
        let end = batch_end.min(Instant::now() + Duration::from_millis(config.proposal_budget_ms));
        let mut quality = BallLocalizationQuality::default();
        let result = refine_one(
            rgb,
            intrinsics,
            proposal,
            &scaled,
            end,
            &mut quality,
            &mut trace,
        );
        let (center, status, reason, outline) = match result {
            Ok((center, outline)) => (
                Some(center),
                RefinementStatus::Accepted,
                None,
                Some(outline),
            ),
            Err(reason) => (None, RefinementStatus::Rejected, Some(reason), None),
        };
        record_trace(&mut trace, proposal.detection_index, &quality, || {
            BallLocalizationAuditEventData::Outcome {
                status,
                reason,
                quality: Box::new(quality.clone()),
                center_image_px: center,
                outline: outline.clone().map(Box::new),
            }
        });
        output.push(BallRefinement {
            detection_index: proposal.detection_index,
            center_image_px: center,
            status,
            reason,
            quality,
            outline,
        });
    }
    // Detector NMS emits proposals in confidence order. Keep bounded work for
    // the configured physical-ball capacity, but retain an explicit rejected
    // result for every extra visible detection so it cannot acquire a stale or
    // guessed precise center.
    for proposal in proposals.iter().skip(config.max_proposals) {
        let quality = BallLocalizationQuality {
            stage: BallFitStage::ProposalCapacity,
            ..BallLocalizationQuality::default()
        };
        record_trace(&mut trace, proposal.detection_index, &quality, || {
            BallLocalizationAuditEventData::Outcome {
                status: RefinementStatus::Rejected,
                reason: Some(RejectionReason::ProposalLimitExceeded),
                quality: Box::new(quality.clone()),
                center_image_px: None,
                outline: None,
            }
        });
        output.push(BallRefinement {
            detection_index: proposal.detection_index,
            center_image_px: None,
            status: RefinementStatus::Rejected,
            reason: Some(RejectionReason::ProposalLimitExceeded),
            quality,
            outline: None,
        });
    }
    Ok(output)
}

fn refine_one(
    rgb: &[u8],
    k: &CameraIntrinsics,
    proposal: &BallProposal,
    cfg: &BallLocalizationConfig,
    end: Instant,
    quality: &mut BallLocalizationQuality,
    trace: &mut Option<&mut BallLocalizationAuditTrace>,
) -> Result<([f64; 2], BallOutline), RejectionReason> {
    deadline(Instant::now(), end)?;
    let [left, top, right, bottom] = proposal.bbox;
    if !proposal.bbox.iter().all(|v| v.is_finite())
        || left < 0.0
        || top < 0.0
        || right > f64::from(k.source_width_px)
        || bottom > f64::from(k.source_height_px)
        || left >= right
        || top >= bottom
    {
        return Err(RejectionReason::InvalidProposal);
    }
    quality.stage = BallFitStage::CropSetup;
    let initial = [(left + right) / 2.0, (top + bottom) / 2.0];
    let half = (right - left).max(bottom - top) * cfg.roi_padding_factor / 2.0;
    if half * 2.0 > f64::from(cfg.max_roi_side_px) {
        return Err(RejectionReason::RoiTooLarge);
    }
    let roi = [
        initial[0] - half,
        initial[1] - half,
        initial[0] + half,
        initial[1] + half,
    ];
    let mut axis = initial;
    let mut final_fit = None;
    for iteration in 0..cfg.max_iterations {
        start_iteration(quality, iteration + 1, Instant::now(), end)?;
        record_trace(trace, proposal.detection_index, quality, || {
            BallLocalizationAuditEventData::IterationStarted {
                axis_image_px: axis,
            }
        });
        quality.stage = BallFitStage::ViewGeometry;
        let view = View::new(k, axis);
        let edges = [
            [left, initial[1]],
            [right, initial[1]],
            [initial[0], top],
            [initial[0], bottom],
        ];
        let mut radii = Vec::new();
        for edge in edges {
            if let Some(p) = view.local(k, edge) {
                radii.push(distance(p, [0.0, 0.0]));
            }
        }
        if radii.len() != 4 {
            return Err(RejectionReason::NumericalFailure);
        }
        let radius = median(&mut radii);
        if radius < cfg.minimum_radius_px || radius > cfg.maximum_radius_px {
            return Err(RejectionReason::InsufficientBoundary);
        }
        let sample = |p: [f64; 2]| {
            let limit = f64::from(cfg.patch_size_px - 1) / 2.0;
            if p[0].abs() > limit || p[1].abs() > limit {
                return None;
            }
            view.source(k, p).and_then(|p| bilinear(rgb, k, p, roi))
        };
        quality.stage = BallFitStage::BackgroundSampling;
        let mut background = [Vec::new(), Vec::new(), Vec::new()];
        for ray in 0..cfg.angular_samples {
            deadline(Instant::now(), end)?;
            let angle = std::f64::consts::TAU * ray as f64 / cfg.angular_samples as f64;
            for factor in [cfg.annulus_inner_factor, cfg.annulus_outer_factor] {
                if let Some(rgb) =
                    sample([angle.cos() * radius * factor, angle.sin() * radius * factor])
                {
                    for ch in 0..3 {
                        background[ch].push(rgb[ch]);
                    }
                }
            }
        }
        let sampled_fraction = background[0].len() as f64 / (2 * cfg.angular_samples) as f64;
        quality.background_sample_fraction = Some(sampled_fraction);
        if sampled_fraction < cfg.minimum_background_supported_fraction {
            return Err(RejectionReason::InsufficientBoundary);
        }
        if background[0].len() < cfg.minimum_inliers {
            return Err(RejectionReason::InsufficientBoundary);
        }
        // Count actual agreement with the robust annulus color, not merely
        // availability of pixels. This rejects a background largely occupied
        // by rail/jaw/neighbor contamination even when samples are in-bounds.
        let original_background = background.clone();
        let bg = [
            median(&mut background[0]),
            median(&mut background[1]),
            median(&mut background[2]),
        ];
        let supported_background = (0..original_background[0].len())
            .filter(|&i| {
                color_distance(
                    [
                        original_background[0][i],
                        original_background[1][i],
                        original_background[2][i],
                    ],
                    bg,
                ) <= cfg.maximum_outer_background_distance
            })
            .count();
        let supported_fraction = supported_background as f64 / (2 * cfg.angular_samples) as f64;
        quality.background_supported_fraction = Some(supported_fraction);
        if supported_fraction < cfg.minimum_background_supported_fraction {
            return Err(RejectionReason::InsufficientBoundary);
        }
        quality.stage = BallFitStage::BoundarySampling;
        let mut candidates = Vec::new();
        for ray in 0..cfg.angular_samples {
            deadline(Instant::now(), end)?;
            let angle = std::f64::consts::TAU * ray as f64 / cfg.angular_samples as f64;
            let direction = [angle.cos(), angle.sin()];
            let mut ray_candidates = Vec::new();
            for step in 0..cfg.radial_samples {
                deadline(Instant::now(), end)?;
                let r = radius
                    * (cfg.radius_min_fraction
                        + (cfg.radius_max_fraction - cfg.radius_min_fraction) * step as f64
                            / (cfg.radial_samples - 1) as f64);
                let at = |r: f64| sample([r * direction[0], r * direction[1]]);
                let (Some(inner), Some(outer), Some(a), Some(b)) = (
                    at(r - cfg.background_probe_px),
                    at(r + cfg.background_probe_px),
                    at(r - cfg.gradient_step_px),
                    at(r + cfg.gradient_step_px),
                ) else {
                    continue;
                };
                let strength = color_distance(a, b);
                if strength < cfg.minimum_gradient
                    || color_distance(inner, bg) < cfg.minimum_foreground_background_contrast
                    || color_distance(outer, bg) > cfg.maximum_outer_background_distance
                    || color_distance(outer, bg) > color_distance(inner, bg)
                {
                    continue;
                }
                ray_candidates.push(Candidate {
                    point: [r * direction[0], r * direction[1]],
                    ray,
                    strength,
                });
            }
            ray_candidates.sort_by(|a, b| b.strength.total_cmp(&a.strength));
            let mut selected = Vec::<Candidate>::new();
            for c in ray_candidates {
                if selected
                    .iter()
                    .all(|p| distance(p.point, c.point) >= cfg.gradient_step_px)
                {
                    selected.push(c);
                    if selected.len() == cfg.candidates_per_ray {
                        break;
                    }
                }
            }
            candidates.extend(selected);
        }
        quality.boundary_candidates = Some(candidates.len());
        let mut supported = vec![false; cfg.angular_samples];
        for c in &candidates {
            supported[c.ray] = true;
        }
        let supported_fraction =
            supported.iter().filter(|v| **v).count() as f64 / cfg.angular_samples as f64;
        quality.supported_ray_fraction = Some(supported_fraction);
        if trace.is_some() {
            for candidate in &candidates {
                deadline(Instant::now(), end)?;
                record_trace(trace, proposal.detection_index, quality, || {
                    BallLocalizationAuditEventData::BoundaryCandidate {
                        ray: candidate.ray,
                        point_local_px: candidate.point,
                        point_image_px: view.source(k, candidate.point),
                        strength: candidate.strength,
                    }
                });
            }
        }
        if supported_fraction < cfg.minimum_supported_ray_fraction {
            return Err(RejectionReason::InsufficientBoundary);
        }
        if candidates.len() < cfg.minimum_inliers {
            return Err(RejectionReason::InsufficientBoundary);
        }
        quality.stage = BallFitStage::HypothesisGeneration;
        // Sampling belongs to the crop and configuration. Detection identity
        // is metadata only and must not change the geometry we estimate.
        let mut seed = cfg.seed;
        let mut best: Option<(Circle, usize, f64)> = None;
        let mut best_id = 0;
        let mut hypotheses = Vec::new();
        for _ in 0..cfg.ransac_iterations {
            deadline(Instant::now(), end)?;
            let a = candidates[next_random(&mut seed, candidates.len())];
            let b = candidates[next_random(&mut seed, candidates.len())];
            let c = candidates[next_random(&mut seed, candidates.len())];
            if a.ray == b.ray || a.ray == c.ray || b.ray == c.ray {
                continue;
            }
            let Some(fit) = three_circle(a.point, b.point, c.point, cfg.numerical_epsilon) else {
                continue;
            };
            if fit.radius < radius * cfg.radius_min_fraction
                || fit.radius > radius * cfg.radius_max_fraction
                || distance(fit.center, [0.0, 0.0]) > radius * cfg.maximum_center_shift_fraction
            {
                continue;
            }
            let selected = select(&candidates, fit, cfg);
            let error = selected
                .iter()
                .map(|c| (distance(c.point, fit.center) - fit.radius).powi(2))
                .sum::<f64>();
            let fit_id = hypotheses.len();
            hypotheses.push((fit, selected.len(), error));
            record_trace(trace, proposal.detection_index, quality, || {
                BallLocalizationAuditEventData::RawHypothesis {
                    fit_id,
                    center_local_px: fit.center,
                    center_image_px: view.source(k, fit.center),
                    radius_local_px: fit.radius,
                    support: selected.len(),
                    squared_error_local_px: error,
                }
            });
            if best.is_none_or(|(_, n, e)| selected.len() > n || (selected.len() == n && error < e))
            {
                best = Some((fit, selected.len(), error));
                best_id = fit_id;
            }
        }
        let Some((fit, best_support, best_error)) = best else {
            return Err(RejectionReason::AmbiguousBoundary);
        };
        quality.fit_id = Some(best_id);
        quality.stage = BallFitStage::AmbiguityAssessment;
        // Competing high-support centers are ambiguity, not extra confidence.
        // Preserve the coarse proposal only when a spatially coherent
        // alternative is repeatedly found. A lone random RANSAC sample can
        // land on another edge combination even when the selected contour's
        // coverage and quadrant-removal stability are strong.
        if has_competing_fit_cluster(&hypotheses, fit, best_support, best_error, cfg, end)? {
            return Err(RejectionReason::AmbiguousBoundary);
        }
        quality.stage = BallFitStage::FitRefinement;
        let selected = select(&candidates, fit, cfg);
        if selected.len() < cfg.minimum_inliers {
            return Err(RejectionReason::InsufficientBoundary);
        }
        let points: Vec<[f64; 2]> = selected.iter().map(|c| c.point).collect();
        let fit = irls(fit, &points, cfg, end)?;
        record_trace(trace, proposal.detection_index, quality, || {
            BallLocalizationAuditEventData::RefinedFit {
                fit_id: best_id,
                center_local_px: fit.center,
                center_image_px: view.source(k, fit.center),
                radius_local_px: fit.radius,
                outline_image_px: (0..cfg.outline_samples)
                    .map(|step| {
                        let angle =
                            std::f64::consts::TAU * step as f64 / cfg.outline_samples as f64;
                        view.source(
                            k,
                            [
                                fit.center[0] + fit.radius * angle.cos(),
                                fit.center[1] + fit.radius * angle.sin(),
                            ],
                        )
                    })
                    .collect(),
            }
        });
        deadline(Instant::now(), end)?;
        let selected = select(&candidates, fit, cfg);
        quality.inliers = Some(selected.len());
        if trace.is_some() {
            for candidate in &selected {
                deadline(Instant::now(), end)?;
                record_trace(trace, proposal.detection_index, quality, || {
                    BallLocalizationAuditEventData::SelectedInlier {
                        fit_id: best_id,
                        ray: candidate.ray,
                        point_local_px: candidate.point,
                        point_image_px: view.source(k, candidate.point),
                    }
                });
            }
        }
        if selected.len() < cfg.minimum_inliers {
            return Err(RejectionReason::InsufficientBoundary);
        }
        quality.stage = BallFitStage::Coverage;
        let (coverage, gap) = coverage(&selected, fit.center, cfg);
        quality.angular_coverage = Some(coverage);
        quality.largest_gap_fraction = Some(gap);
        if coverage < cfg.minimum_coverage || gap > cfg.maximum_gap_fraction {
            return Err(RejectionReason::InadequateCoverage);
        }
        quality.stage = BallFitStage::Residual;
        let rms = (selected
            .iter()
            .map(|c| (distance(c.point, fit.center) - fit.radius).powi(2))
            .sum::<f64>()
            / selected.len() as f64)
            .sqrt();
        quality.residual_rms_px = Some(rms);
        if rms > cfg.maximum_residual_px {
            return Err(RejectionReason::ExcessiveResidual);
        }
        quality.stage = BallFitStage::CenterProjection;
        let source = view
            .source(k, fit.center)
            .ok_or(RejectionReason::NumericalFailure)?;
        let shift = distance(source, initial);
        quality.center_shift_px = Some(shift);
        if shift > (right - left).min(bottom - top) / 2.0 * cfg.maximum_center_shift_fraction {
            return Err(RejectionReason::ExcessiveCenterShift);
        }
        // Leave-sector-group-out fits expose arc sensitivity without pretending
        // this spread is a calibrated statistical uncertainty or confidence.
        quality.stage = BallFitStage::CenterStability;
        let mut spread: f64 = 0.0;
        for omit in 0..4 {
            deadline(Instant::now(), end)?;
            let subset: Vec<[f64; 2]> = selected
                .iter()
                .filter(|c| {
                    let a = (c.point[1] - fit.center[1])
                        .atan2(c.point[0] - fit.center[0])
                        .rem_euclid(std::f64::consts::TAU);
                    (a / std::f64::consts::FRAC_PI_2).floor() as usize != omit
                })
                .map(|c| c.point)
                .collect();
            if subset.len() < 3 {
                return Err(RejectionReason::UnstableCenter);
            }
            let other = irls(fit, &subset, cfg, end)?;
            let other = view
                .source(k, other.center)
                .ok_or(RejectionReason::NumericalFailure)?;
            spread = spread.max(distance(source, other));
        }
        quality.center_spread_px = Some(spread);
        if spread > cfg.maximum_center_spread_px {
            return Err(RejectionReason::UnstableCenter);
        }
        quality.stage = BallFitStage::Foreground;
        let mut foreground = 0;
        let mut available = 0;
        // Probe the interior across all ray directions, rather than accepting
        // a small bright number or highlight as an entire ball silhouette.
        for ray in 0..cfg.angular_samples {
            deadline(Instant::now(), end)?;
            let angle = std::f64::consts::TAU * ray as f64 / cfg.angular_samples as f64;
            let r = fit.radius * cfg.radius_min_fraction;
            if let Some(p) = sample([
                fit.center[0] + r * angle.cos(),
                fit.center[1] + r * angle.sin(),
            ]) {
                available += 1;
                if color_distance(p, bg) >= cfg.minimum_foreground_background_contrast {
                    foreground += 1;
                }
            }
        }
        let fraction = foreground as f64 / available.max(1) as f64;
        quality.foreground_fraction = Some(fraction);
        if fraction < cfg.minimum_foreground_fraction {
            return Err(RejectionReason::InsufficientForeground);
        }
        let change = distance(axis, source);
        axis = source;
        final_fit = Some((view, fit, source));
        if change < cfg.convergence_px {
            break;
        }
    }
    quality.stage = BallFitStage::OutlineProjection;
    deadline(Instant::now(), end)?;
    let (view, fit, center) = final_fit.ok_or(RejectionReason::NumericalFailure)?;
    let outline = project_outline(k, view, fit, center, cfg, end, quality)?;
    Ok((center, outline))
}

fn project_outline(
    k: &CameraIntrinsics,
    view: View,
    fit: Circle,
    center: [f64; 2],
    cfg: &BallLocalizationConfig,
    end: Instant,
    quality: &mut BallLocalizationQuality,
) -> Result<BallOutline, RejectionReason> {
    quality.stage = BallFitStage::OutlineProjection;
    deadline(Instant::now(), end)?;
    let mut points = Vec::new();
    let mut extent = [0.0f64; 2];
    for step in 0..cfg.outline_samples {
        deadline(Instant::now(), end)?;
        let angle = std::f64::consts::TAU * step as f64 / cfg.outline_samples as f64;
        let p = view
            .source(
                k,
                [
                    fit.center[0] + fit.radius * angle.cos(),
                    fit.center[1] + fit.radius * angle.sin(),
                ],
            )
            .ok_or(RejectionReason::NumericalFailure)?;
        if p[0] < 0.0
            || p[1] < 0.0
            || p[0] >= f64::from(k.source_width_px)
            || p[1] >= f64::from(k.source_height_px)
        {
            return Err(RejectionReason::InadequateCoverage);
        }
        for i in 0..2 {
            extent[i] = extent[i].max((p[i] - center[i]).abs());
        }
        points.push(p);
    }
    quality.stage = BallFitStage::Complete;
    Ok(BallOutline {
        points_image_px: points,
        half_extent_image_px: extent,
        local_radius_px: fit.radius,
    })
}

#[cfg(test)]
mod diagnostic_tests {
    use super::{
        project_outline, start_iteration, BallFitStage, BallLocalizationAuditEvent,
        BallLocalizationAuditEventData, BallLocalizationAuditTrace, BallLocalizationConfig,
        BallLocalizationQuality, CameraIntrinsics, Circle, RejectionReason, View,
        BALL_LOCALIZATION_AUDIT_VERSION,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn deadline_at_new_iteration_does_not_retain_previous_fit_metrics() {
        let mut quality = BallLocalizationQuality {
            stage: BallFitStage::CenterStability,
            fit_id: Some(9),
            iterations: 1,
            boundary_candidates: Some(92),
            inliers: Some(80),
            residual_rms_px: Some(0.2),
            center_spread_px: Some(0.1),
            ..BallLocalizationQuality::default()
        };
        let now = Instant::now();
        assert_eq!(
            start_iteration(&mut quality, 2, now, now - Duration::from_millis(1)),
            Err(RejectionReason::DeadlineExceeded)
        );
        assert_eq!(quality.iterations, 2);
        assert_eq!(quality.stage, BallFitStage::IterationStart);
        assert_eq!(quality.fit_id, None);
        assert_eq!(quality.boundary_candidates, None);
        assert_eq!(quality.inliers, None);
        assert_eq!(quality.residual_rms_px, None);
        assert_eq!(quality.center_spread_px, None);
    }

    #[test]
    fn outline_failure_is_attributed_to_outline_projection() {
        let k = CameraIntrinsics {
            source_width_px: 480,
            source_height_px: 300,
            fx: 260.0,
            fy: 273.0,
            cx: 240.0,
            cy: 150.0,
        };
        let center = [1.0, 1.0];
        let view = View::new(&k, center);
        let fit = Circle {
            center: [0.0, 0.0],
            radius: 20.0,
        };
        let mut quality = BallLocalizationQuality {
            stage: BallFitStage::Foreground,
            iterations: 2,
            residual_rms_px: Some(0.2),
            ..BallLocalizationQuality::default()
        };
        let result = project_outline(
            &k,
            view,
            fit,
            center,
            &BallLocalizationConfig::default(),
            Instant::now() + Duration::from_secs(1),
            &mut quality,
        );
        assert_eq!(result, Err(RejectionReason::InadequateCoverage));
        assert_eq!(quality.stage, BallFitStage::OutlineProjection);
        assert_eq!(quality.iterations, 2);
        assert_eq!(quality.residual_rms_px, Some(0.2));
    }

    #[test]
    fn audit_drops_variable_payload_when_byte_budget_is_exhausted() {
        let events = Vec::<BallLocalizationAuditEvent>::with_capacity(1);
        let reserved = events.capacity() * std::mem::size_of::<BallLocalizationAuditEvent>();
        let mut trace = BallLocalizationAuditTrace {
            schema_version: BALL_LOCALIZATION_AUDIT_VERSION,
            maximum_events: 1,
            maximum_bytes: reserved + 1,
            retained_bytes: reserved,
            dropped_events: 0,
            events,
        };
        trace.record(|| BallLocalizationAuditEvent {
            detection_index: 7,
            iteration: Some(1),
            stage: BallFitStage::FitRefinement,
            data: BallLocalizationAuditEventData::RefinedFit {
                fit_id: 0,
                center_local_px: [0.0, 0.0],
                center_image_px: Some([1.0, 1.0]),
                radius_local_px: 10.0,
                outline_image_px: vec![Some([1.0, 1.0]); 3],
            },
        });
        assert!(trace.events.is_empty());
        assert_eq!(trace.dropped_events, 1);
        assert_eq!(trace.retained_bytes, reserved);
    }
}

#[cfg(test)]
mod ambiguity_tests {
    use super::{has_competing_fit_cluster, BallLocalizationConfig, Circle};
    use std::time::{Duration, Instant};

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(1)
    }

    #[test]
    fn isolated_alternative_ransac_sample_does_not_reject_a_stable_center() {
        let config = BallLocalizationConfig::default();
        let best = Circle {
            center: [0.0, 0.0],
            radius: 20.0,
        };
        let hypotheses = [
            (best, 96, 1.0),
            (
                Circle {
                    center: [4.0, 0.0],
                    radius: 20.1,
                },
                95,
                1.05,
            ),
        ];

        assert!(
            !has_competing_fit_cluster(&hypotheses, best, 96, 1.0, &config, deadline()).unwrap()
        );
    }

    #[test]
    fn repeated_plausible_alternative_center_is_rejected() {
        let config = BallLocalizationConfig::default();
        let best = Circle {
            center: [0.0, 0.0],
            radius: 20.0,
        };
        let alternative = Circle {
            center: [4.0, 0.0],
            radius: 20.1,
        };
        let hypotheses = [
            (best, 96, 1.0),
            (alternative, 95, 1.05),
            (alternative, 95, 1.06),
            (alternative, 94, 1.07),
        ];

        assert!(
            has_competing_fit_cluster(&hypotheses, best, 96, 1.0, &config, deadline()).unwrap()
        );
    }

    #[test]
    fn competing_fit_scan_respects_the_refinement_deadline() {
        let config = BallLocalizationConfig::default();
        let best = Circle {
            center: [0.0, 0.0],
            radius: 20.0,
        };
        let hypotheses = [(best, 96, 1.0)];
        let expired = Instant::now() - Duration::from_millis(1);

        assert_eq!(
            has_competing_fit_cluster(&hypotheses, best, 96, 1.0, &config, expired),
            Err(super::RejectionReason::DeadlineExceeded)
        );
    }
}
