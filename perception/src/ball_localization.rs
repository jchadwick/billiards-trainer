//! Bounded, frame-local 2D ball localization on borrowed native RGB pixels.
//!
//! A caller supplies the intrinsics bound to the actual source geometry. Local
//! camera-ray rotation removes apparent off-axis stretching without estimating
//! ball depth, changing physical diameter, or modifying detector boxes. Every
//! proposal receives an explicit accepted/rejected result; fit diagnostics are
//! sensitivity measures, never calibrated confidence or physical uncertainty.

mod cyclic;

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
    /// Maximum number of geometrically diverse, plausible candidates that
    /// may be fully refined for one aiming iteration.
    pub maximum_refined_candidates: usize,
    /// Maximum rounds in which a refined fit is reselected against radial
    /// boundary candidates and fit again.
    pub maximum_inlier_reselection_rounds: usize,
    /// Competing, fully qualified candidates must have residual RMS no worse
    /// than this factor of the best candidate, plus the configured noise floor.
    pub maximum_alternative_error_ratio: f64,
    /// Additive source-pixel noise floor for comparing alternative residuals.
    pub maximum_alternative_residual_tolerance_px: f64,
    /// Source-pixel outline difference above which two candidates cannot be
    /// treated as the same visible boundary.
    pub maximum_same_boundary_outline_disagreement_px: f64,
    pub minimum_background_supported_fraction: f64,
    pub batch_budget_ms: u64,
    pub proposal_budget_ms: u64,
    pub max_roi_side_px: u32,
    pub roi_padding_factor: f64,
    pub angular_samples: usize,
    pub radial_samples: usize,
    pub candidates_per_ray: usize,
    pub max_iterations: usize,
    pub maximum_cyclic_paths: usize,
    pub cyclic_continuity_weight: f64,
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
}
impl Default for BallLocalizationConfig {
    fn default() -> Self {
        Self {
            max_proposals: 16,
            reference_width_px: 3840,
            patch_size_px: 193,
            minimum_supported_ray_fraction: 0.5,
            minimum_alternative_support_fraction: 0.95,
            maximum_refined_candidates: 16,
            maximum_inlier_reselection_rounds: 3,
            maximum_alternative_error_ratio: 1.1,
            maximum_alternative_residual_tolerance_px: 0.25,
            maximum_same_boundary_outline_disagreement_px: 1.5,
            minimum_background_supported_fraction: 0.6,
            batch_budget_ms: 25,
            proposal_budget_ms: 10,
            max_roi_side_px: 512,
            roi_padding_factor: 1.8,
            angular_samples: 96,
            radial_samples: 96,
            candidates_per_ray: 3,
            max_iterations: 3,
            maximum_cyclic_paths: 8,
            cyclic_continuity_weight: 1.0,
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
            self.maximum_alternative_residual_tolerance_px,
            self.maximum_same_boundary_outline_disagreement_px,
            self.irls_tuning,
            self.irls_scale_floor_px,
            self.convergence_px,
            self.numerical_epsilon,
            self.cyclic_continuity_weight,
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
            || self.maximum_cyclic_paths > 16
            || self.maximum_refined_candidates == 0
            || self.maximum_refined_candidates > 16
            || self.maximum_inlier_reselection_rounds == 0
            || self.maximum_inlier_reselection_rounds > 3
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
            || self.maximum_cyclic_paths == 0
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
    RadiusOutOfRange,
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
    pub candidate_outer_background_fraction: Option<f64>,
    pub inliers: Option<usize>,
    pub angular_coverage: Option<f64>,
    pub largest_gap_fraction: Option<f64>,
    pub residual_rms_px: Option<f64>,
    /// Radial fit RMS after mapping measured edge points and fitted boundary
    /// points back into the native source image.
    pub source_residual_rms_px: Option<f64>,
    pub center_shift_px: Option<f64>,
    pub center_spread_px: Option<f64>,
    pub foreground_fraction: Option<f64>,
    pub iterations: usize,
    pub candidate_representatives_evaluated: usize,
    pub qualified_candidate_count: usize,
    pub candidate_search_truncated: bool,
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
    CandidateSelection,
    AmbiguityAssessment,
    FitRefinement,
    CandidateQualification,
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
    CyclicHypothesis {
        fit_id: usize,
        path_cost: f64,
        candidate_indices: Vec<usize>,
        center_local_px: [f64; 2],
        center_image_px: Option<[f64; 2]>,
        radius_local_px: f64,
        support: usize,
        source_residual_rms_px: f64,
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
    CandidateEvaluation {
        fit_id: usize,
        center_image_px: Option<[f64; 2]>,
        supported_rays: usize,
        residual_rms_px: Option<f64>,
        source_residual_rms_px: Option<f64>,
        rejection: Option<RejectionReason>,
        quality: Box<BallLocalizationQuality>,
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
pub const BALL_LOCALIZATION_AUDIT_VERSION: u32 = 4;

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
            BallLocalizationAuditEventData::CandidateEvaluation {
                quality: _,
                outline_image_px,
                ..
            } => {
                std::mem::size_of::<BallLocalizationQuality>()
                    + outline_image_px.capacity() * std::mem::size_of::<Option<[f64; 2]>>()
            }
            BallLocalizationAuditEventData::Outcome { outline: None, .. } => {
                std::mem::size_of::<BallLocalizationQuality>()
            }
            BallLocalizationAuditEventData::CyclicHypothesis {
                candidate_indices, ..
            } => candidate_indices.capacity() * std::mem::size_of::<usize>(),
            BallLocalizationAuditEventData::IterationStarted { .. }
            | BallLocalizationAuditEventData::BoundaryCandidate { .. }
            | BallLocalizationAuditEventData::SelectedInlier { .. } => 0,
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

fn strongest_distinct_ray_transitions(
    mut candidates: Vec<Candidate>,
    maximum_count: usize,
    minimum_separation_px: f64,
) -> Vec<Candidate> {
    candidates.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    let mut selected = Vec::<Candidate>::new();
    for candidate in candidates {
        if selected
            .iter()
            .all(|previous| distance(previous.point, candidate.point) >= minimum_separation_px)
        {
            selected.push(candidate);
            if selected.len() == maximum_count {
                break;
            }
        }
    }
    selected
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Circle {
    center: [f64; 2],
    radius: f64,
}

#[derive(Clone, Debug)]
struct RawHypothesis {
    fit_id: usize,
    fit: Circle,
    supported_rays: usize,
    source_residual_rms_px: f64,
    center_image_px: [f64; 2],
}

#[derive(Clone, Debug)]
struct CandidateRepresentative {
    hypothesis: RawHypothesis,
    outline_image_px: Vec<[f64; 2]>,
}

#[derive(Debug)]
struct CandidateAttempt {
    fit_id: usize,
    fit: Circle,
    supported_rays: usize,
    source_residual_rms_px: Option<f64>,
    center_image_px: Option<[f64; 2]>,
    outline: Option<BallOutline>,
    diagnostic_outline_image_px: Vec<Option<[f64; 2]>>,
    quality: BallLocalizationQuality,
    rejection: Option<RejectionReason>,
}

#[derive(Clone, Copy)]
struct CandidateEvaluationContext<'a> {
    candidates: &'a [Candidate],
    view: View,
    intrinsics: &'a CameraIntrinsics,
    initial_center_image_px: [f64; 2],
    proposal_radius_px: f64,
    bbox_width_px: f64,
    bbox_height_px: f64,
    background_rgb: [f64; 3],
    sample: &'a dyn Fn([f64; 2]) -> Option<[f64; 3]>,
    config: &'a BallLocalizationConfig,
    deadline: Instant,
    base_quality: &'a BallLocalizationQuality,
    detection_index: u64,
}

impl CandidateAttempt {
    fn is_qualified(&self) -> bool {
        self.rejection.is_none() && self.center_image_px.is_some() && self.outline.is_some()
    }
}

fn hypothesis_rank(a: &RawHypothesis, b: &RawHypothesis) -> std::cmp::Ordering {
    b.supported_rays
        .cmp(&a.supported_rays)
        .then_with(|| {
            a.source_residual_rms_px
                .total_cmp(&b.source_residual_rms_px)
        })
        .then_with(|| a.center_image_px[0].total_cmp(&b.center_image_px[0]))
        .then_with(|| a.center_image_px[1].total_cmp(&b.center_image_px[1]))
        .then_with(|| a.fit.radius.total_cmp(&b.fit.radius))
        .then_with(|| a.fit_id.cmp(&b.fit_id))
}

fn residual_is_plausible(
    support: usize,
    residual_rms_px: f64,
    best_support: usize,
    best_residual_rms_px: f64,
    cfg: &BallLocalizationConfig,
) -> bool {
    support as f64 >= best_support as f64 * cfg.minimum_alternative_support_fraction
        && residual_rms_px
            <= best_residual_rms_px * cfg.maximum_alternative_error_ratio
                + cfg.maximum_alternative_residual_tolerance_px
}

fn outline_disagreement_px(
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    end: Instant,
) -> Result<f64, RejectionReason> {
    fn directed_max_distance(
        source: &[[f64; 2]],
        target: &[[f64; 2]],
        end: Instant,
    ) -> Result<f64, RejectionReason> {
        let mut greatest = 0.0f64;
        for point in source {
            deadline(Instant::now(), end)?;
            let mut nearest = f64::INFINITY;
            for other in target {
                deadline(Instant::now(), end)?;
                nearest = nearest.min(distance(*point, *other));
            }
            greatest = greatest.max(nearest);
        }
        Ok(greatest)
    }
    if a.is_empty() || b.is_empty() {
        return Ok(f64::INFINITY);
    }
    Ok(directed_max_distance(a, b, end)?.max(directed_max_distance(b, a, end)?))
}

fn same_source_boundary(
    a_center: [f64; 2],
    a_outline: &[[f64; 2]],
    b_center: [f64; 2],
    b_outline: &[[f64; 2]],
    cfg: &BallLocalizationConfig,
    end: Instant,
) -> Result<bool, RejectionReason> {
    if distance(a_center, b_center) > cfg.maximum_center_spread_px {
        return Ok(false);
    }
    Ok(outline_disagreement_px(a_outline, b_outline, end)?
        <= cfg.maximum_same_boundary_outline_disagreement_px)
}

fn source_outline_points(
    k: &CameraIntrinsics,
    view: View,
    fit: Circle,
    sample_count: usize,
    end: Instant,
) -> Result<Vec<[f64; 2]>, RejectionReason> {
    let mut points = Vec::with_capacity(sample_count);
    for step in 0..sample_count {
        deadline(Instant::now(), end)?;
        let angle = std::f64::consts::TAU * step as f64 / sample_count as f64;
        points.push(
            view.source(
                k,
                [
                    fit.center[0] + fit.radius * angle.cos(),
                    fit.center[1] + fit.radius * angle.sin(),
                ],
            )
            .ok_or(RejectionReason::NumericalFailure)?,
        );
    }
    Ok(points)
}

/// Measure residuals in native source pixels, where configured comparison
/// noise and outline tolerances are defined. A local-patch residual is not
/// interchangeable with a source-pixel residual because the inverse view
/// projection has position-dependent magnification.
fn source_residual_rms_px(
    candidates: &[Candidate],
    fit: Circle,
    view: View,
    k: &CameraIntrinsics,
    end: Instant,
) -> Result<f64, RejectionReason> {
    if candidates.is_empty() {
        return Err(RejectionReason::InsufficientBoundary);
    }
    let mut squared_error = 0.0;
    for candidate in candidates {
        deadline(Instant::now(), end)?;
        let dx = candidate.point[0] - fit.center[0];
        let dy = candidate.point[1] - fit.center[1];
        let distance_local = dx.hypot(dy);
        if distance_local <= f64::EPSILON {
            return Err(RejectionReason::NumericalFailure);
        }
        let boundary_local = [
            fit.center[0] + fit.radius * dx / distance_local,
            fit.center[1] + fit.radius * dy / distance_local,
        ];
        let observed_source = view
            .source(k, candidate.point)
            .ok_or(RejectionReason::NumericalFailure)?;
        let boundary_source = view
            .source(k, boundary_local)
            .ok_or(RejectionReason::NumericalFailure)?;
        squared_error += distance(observed_source, boundary_source).powi(2);
    }
    Ok((squared_error / candidates.len() as f64).sqrt())
}

fn choose_diverse_representatives(
    hypotheses: &[RawHypothesis],
    k: &CameraIntrinsics,
    view: View,
    cfg: &BallLocalizationConfig,
    end: Instant,
) -> Result<(Vec<CandidateRepresentative>, bool), RejectionReason> {
    let mut ranked = hypotheses.to_vec();
    ranked.sort_by(hypothesis_rank);
    let Some(best) = ranked.first() else {
        return Ok((Vec::new(), false));
    };
    let best_support = best.supported_rays;
    let best_residual_rms_px = best.source_residual_rms_px;
    let mut representatives: Vec<CandidateRepresentative> = Vec::new();
    let mut truncated = false;
    for hypothesis in ranked {
        deadline(Instant::now(), end)?;
        if !residual_is_plausible(
            hypothesis.supported_rays,
            hypothesis.source_residual_rms_px,
            best_support,
            best_residual_rms_px,
            cfg,
        ) {
            continue;
        }
        let mut duplicate = false;
        for representative in &representatives {
            if distance(
                hypothesis.center_image_px,
                representative.hypothesis.center_image_px,
            ) > cfg.maximum_center_spread_px
            {
                continue;
            }
            let outline = source_outline_points(k, view, hypothesis.fit, cfg.outline_samples, end)?;
            if same_source_boundary(
                hypothesis.center_image_px,
                &outline,
                representative.hypothesis.center_image_px,
                &representative.outline_image_px,
                cfg,
                end,
            )? {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            continue;
        }
        if representatives.len() == cfg.maximum_refined_candidates {
            truncated = true;
            continue;
        }
        let outline = source_outline_points(k, view, hypothesis.fit, cfg.outline_samples, end)?;
        representatives.push(CandidateRepresentative {
            hypothesis,
            outline_image_px: outline,
        });
    }
    Ok((representatives, truncated))
}

fn same_inlier_set(a: &[Candidate], b: &[Candidate]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(left, right)| {
            left.ray == right.ray && left.point == right.point && left.strength == right.strength
        })
}

fn candidate_attempt_rank(a: &CandidateAttempt, b: &CandidateAttempt) -> std::cmp::Ordering {
    b.supported_rays
        .cmp(&a.supported_rays)
        .then_with(|| {
            a.source_residual_rms_px
                .unwrap_or(f64::INFINITY)
                .total_cmp(&b.source_residual_rms_px.unwrap_or(f64::INFINITY))
        })
        .then_with(|| {
            a.center_image_px.unwrap_or([f64::INFINITY; 2])[0]
                .total_cmp(&b.center_image_px.unwrap_or([f64::INFINITY; 2])[0])
        })
        .then_with(|| {
            a.center_image_px.unwrap_or([f64::INFINITY; 2])[1]
                .total_cmp(&b.center_image_px.unwrap_or([f64::INFINITY; 2])[1])
        })
        .then_with(|| a.fit.radius.total_cmp(&b.fit.radius))
        .then_with(|| a.fit_id.cmp(&b.fit_id))
}

fn within_center_shift_budget(
    displacement_source_px: f64,
    box_width_source_px: f64,
    box_height_source_px: f64,
    maximum_shift_fraction: f64,
) -> bool {
    let maximum_shift = box_width_source_px.min(box_height_source_px) * maximum_shift_fraction;
    displacement_source_px <= maximum_shift
}

fn within_refined_radius_bounds(
    radius_px: f64,
    proposal_radius_px: f64,
    cfg: &BallLocalizationConfig,
) -> bool {
    radius_px.is_finite()
        && radius_px >= cfg.minimum_radius_px
        && radius_px <= cfg.maximum_radius_px
        && radius_px >= proposal_radius_px * cfg.radius_min_fraction
        && radius_px <= proposal_radius_px * cfg.radius_max_fraction
}

fn has_distinct_qualified_alternative(
    attempts: &[CandidateAttempt],
    qualified_indices: &[usize],
    cfg: &BallLocalizationConfig,
    end: Instant,
) -> Result<bool, RejectionReason> {
    let Some(&best_index) = qualified_indices.first() else {
        return Ok(false);
    };
    let best = &attempts[best_index];
    let best_center = best.center_image_px.expect("qualified candidate center");
    let best_outline = best.outline.as_ref().expect("qualified candidate outline");
    let best_residual = best
        .source_residual_rms_px
        .expect("qualified candidate residual");
    for &candidate_index in qualified_indices.iter().skip(1) {
        deadline(Instant::now(), end)?;
        let alternative = &attempts[candidate_index];
        let alternative_residual = alternative
            .source_residual_rms_px
            .expect("qualified alternative residual");
        if !residual_is_plausible(
            alternative.supported_rays,
            alternative_residual,
            best.supported_rays,
            best_residual,
            cfg,
        ) {
            continue;
        }
        let alternative_center = alternative
            .center_image_px
            .expect("qualified alternative center");
        let alternative_outline = alternative
            .outline
            .as_ref()
            .expect("qualified alternative outline");
        if !same_source_boundary(
            best_center,
            &best_outline.points_image_px,
            alternative_center,
            &alternative_outline.points_image_px,
            cfg,
            end,
        )? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn evaluate_candidate(
    raw: &RawHypothesis,
    context: &CandidateEvaluationContext<'_>,
    trace: &mut Option<&mut BallLocalizationAuditTrace>,
) -> CandidateAttempt {
    let CandidateEvaluationContext {
        candidates,
        view,
        intrinsics: k,
        initial_center_image_px,
        proposal_radius_px,
        bbox_width_px,
        bbox_height_px,
        background_rgb,
        sample,
        config: cfg,
        deadline: end,
        base_quality,
        detection_index,
    } = *context;
    let mut fit = raw.fit;
    let mut supported_rays = raw.supported_rays;
    let mut source_rms_px = None;
    let mut center_image_px = None;
    let mut outline = None;
    let mut diagnostic_outline_image_px = Vec::new();
    let mut quality = base_quality.clone();
    quality.fit_id = Some(raw.fit_id);
    quality.stage = BallFitStage::FitRefinement;

    let evaluation = (|| -> Result<(), RejectionReason> {
        let mut selected = select(candidates, fit, cfg);
        if selected.len() < cfg.minimum_inliers {
            return Err(RejectionReason::InsufficientBoundary);
        }
        let mut converged = false;
        for _round in 0..cfg.maximum_inlier_reselection_rounds {
            deadline(Instant::now(), end)?;
            let previous = selected;
            let points: Vec<[f64; 2]> = previous.iter().map(|candidate| candidate.point).collect();
            fit = irls(fit, &points, cfg, end)?;
            selected = select(candidates, fit, cfg);
            if selected.len() < cfg.minimum_inliers {
                return Err(RejectionReason::InsufficientBoundary);
            }
            if same_inlier_set(&previous, &selected) {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(RejectionReason::AmbiguousBoundary);
        }
        quality.stage = BallFitStage::CandidateQualification;
        if !within_refined_radius_bounds(fit.radius, proposal_radius_px, cfg) {
            return Err(RejectionReason::RadiusOutOfRange);
        }
        supported_rays = selected.len();
        quality.inliers = Some(supported_rays);
        let center = view
            .source(k, fit.center)
            .ok_or(RejectionReason::NumericalFailure)?;
        center_image_px = Some(center);
        let projected = source_outline_points(k, view, fit, cfg.outline_samples, end)?;
        diagnostic_outline_image_px = projected.iter().copied().map(Some).collect();

        if trace.is_some() {
            for candidate in &selected {
                deadline(Instant::now(), end)?;
                record_trace(trace, detection_index, &quality, || {
                    BallLocalizationAuditEventData::SelectedInlier {
                        fit_id: raw.fit_id,
                        ray: candidate.ray,
                        point_local_px: candidate.point,
                        point_image_px: view.source(k, candidate.point),
                    }
                });
            }
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
            .map(|candidate| (distance(candidate.point, fit.center) - fit.radius).powi(2))
            .sum::<f64>()
            / selected.len() as f64)
            .sqrt();
        quality.residual_rms_px = Some(rms);
        if rms > cfg.maximum_residual_px {
            return Err(RejectionReason::ExcessiveResidual);
        }
        let source_rms = source_residual_rms_px(&selected, fit, view, k, end)?;
        source_rms_px = Some(source_rms);
        quality.source_residual_rms_px = Some(source_rms);

        let mut outer_available = 0usize;
        let mut outer_supported = 0usize;
        for ray in 0..cfg.angular_samples {
            deadline(Instant::now(), end)?;
            let angle = std::f64::consts::TAU * ray as f64 / cfg.angular_samples as f64;
            let point = [
                fit.center[0] + (fit.radius + cfg.background_probe_px) * angle.cos(),
                fit.center[1] + (fit.radius + cfg.background_probe_px) * angle.sin(),
            ];
            if let Some(color) = sample(point) {
                outer_available += 1;
                if color_distance(color, background_rgb) <= cfg.maximum_outer_background_distance {
                    outer_supported += 1;
                }
            }
        }
        let outer_sample_fraction = outer_available as f64 / cfg.angular_samples as f64;
        let outer_supported_fraction = outer_supported as f64 / outer_available.max(1) as f64;
        quality.candidate_outer_background_fraction = Some(outer_supported_fraction);
        if outer_sample_fraction < cfg.minimum_background_supported_fraction
            || outer_supported_fraction < cfg.minimum_background_supported_fraction
        {
            return Err(RejectionReason::InsufficientBoundary);
        }

        quality.stage = BallFitStage::CenterProjection;
        let shift = distance(center, initial_center_image_px);
        quality.center_shift_px = Some(shift);
        if !within_center_shift_budget(
            shift,
            bbox_width_px,
            bbox_height_px,
            cfg.maximum_center_shift_fraction,
        ) {
            return Err(RejectionReason::ExcessiveCenterShift);
        }

        quality.stage = BallFitStage::CenterStability;
        let mut spread = 0.0f64;
        for omit in 0..4 {
            deadline(Instant::now(), end)?;
            let subset: Vec<[f64; 2]> = selected
                .iter()
                .filter(|candidate| {
                    let angle = (candidate.point[1] - fit.center[1])
                        .atan2(candidate.point[0] - fit.center[0])
                        .rem_euclid(std::f64::consts::TAU);
                    (angle / std::f64::consts::FRAC_PI_2).floor() as usize != omit
                })
                .map(|candidate| candidate.point)
                .collect();
            if subset.len() < 3 {
                return Err(RejectionReason::UnstableCenter);
            }
            let other = irls(fit, &subset, cfg, end)?;
            if !within_refined_radius_bounds(other.radius, proposal_radius_px, cfg) {
                return Err(RejectionReason::RadiusOutOfRange);
            }
            let other = view
                .source(k, other.center)
                .ok_or(RejectionReason::NumericalFailure)?;
            spread = spread.max(distance(center, other));
        }
        quality.center_spread_px = Some(spread);
        if spread > cfg.maximum_center_spread_px {
            return Err(RejectionReason::UnstableCenter);
        }

        quality.stage = BallFitStage::Foreground;
        let mut foreground = 0usize;
        let mut foreground_available = 0usize;
        for ray in 0..cfg.angular_samples {
            deadline(Instant::now(), end)?;
            let angle = std::f64::consts::TAU * ray as f64 / cfg.angular_samples as f64;
            let radius = fit.radius * cfg.radius_min_fraction;
            if let Some(color) = sample([
                fit.center[0] + radius * angle.cos(),
                fit.center[1] + radius * angle.sin(),
            ]) {
                foreground_available += 1;
                if color_distance(color, background_rgb)
                    >= cfg.minimum_foreground_background_contrast
                {
                    foreground += 1;
                }
            }
        }
        let foreground_fraction = foreground as f64 / foreground_available.max(1) as f64;
        quality.foreground_fraction = Some(foreground_fraction);
        if foreground_fraction < cfg.minimum_foreground_fraction {
            return Err(RejectionReason::InsufficientForeground);
        }

        outline = Some(outline_from_source_points(
            k,
            fit,
            center,
            &projected,
            end,
            &mut quality,
        )?);
        Ok(())
    })();

    let rejection = evaluation.err();
    CandidateAttempt {
        fit_id: raw.fit_id,
        fit,
        supported_rays,
        source_residual_rms_px: source_rms_px,
        center_image_px,
        outline,
        diagnostic_outline_image_px,
        quality,
        rejection,
    }
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
    scaled.maximum_alternative_residual_tolerance_px *= scale;
    scaled.maximum_same_boundary_outline_disagreement_px *= scale;
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
            // Adjacent samples whose gradient footprints overlap are the same
            // observed transition, not independent contour hypotheses. Keep
            // genuinely separated transitions on the same ray.
            candidates.extend(strongest_distinct_ray_transitions(
                ray_candidates,
                cfg.candidates_per_ray,
                2.0 * cfg.gradient_step_px,
            ));
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
        // A complete cyclic contour uses global angular continuity, rather
        // than independent random triplets. Pixel qualification below is still
        // required: minimum objective cost does not identify physical truth.
        let mut hypotheses = Vec::new();
        let mut hypothesis_paths: Vec<Vec<Candidate>> = Vec::new();
        for path in cyclic::paths(&candidates, cfg, end)? {
            deadline(Instant::now(), end)?;
            let points: Vec<_> = path.indices.iter().map(|&i| candidates[i].point).collect();
            let Some(initial_fit) = cyclic::initialize(&points, cfg.numerical_epsilon) else {
                continue;
            };
            let fit = irls(initial_fit, &points, cfg, end)?;
            if !within_refined_radius_bounds(fit.radius, radius, cfg) {
                continue;
            }
            let Some(center_image_px) = view.source(k, fit.center) else {
                continue;
            };
            if !within_center_shift_budget(
                distance(center_image_px, initial),
                right - left,
                bottom - top,
                cfg.maximum_center_shift_fraction,
            ) {
                continue;
            }
            let path_candidates: Vec<_> = path
                .indices
                .iter()
                .map(|&index| candidates[index])
                .collect();
            let selected = select(&path_candidates, fit, cfg);
            if selected.len() < cfg.minimum_inliers {
                continue;
            }
            let source_residual_rms_px = source_residual_rms_px(&selected, fit, view, k, end)?;
            let fit_id = hypotheses.len();
            hypotheses.push(RawHypothesis {
                fit_id,
                fit,
                supported_rays: selected.len(),
                source_residual_rms_px,
                center_image_px,
            });
            hypothesis_paths.push(path_candidates);
            record_trace(trace, proposal.detection_index, quality, || {
                BallLocalizationAuditEventData::CyclicHypothesis {
                    fit_id,
                    path_cost: path.cost,
                    candidate_indices: path.indices,
                    center_local_px: fit.center,
                    center_image_px: Some(center_image_px),
                    radius_local_px: fit.radius,
                    support: selected.len(),
                    source_residual_rms_px,
                }
            });
        }
        quality.stage = BallFitStage::CandidateSelection;
        let (representatives, search_truncated) =
            choose_diverse_representatives(&hypotheses, k, view, cfg, end)?;
        if representatives.is_empty() {
            quality.candidate_search_truncated = search_truncated;
            return Err(if search_truncated {
                RejectionReason::AmbiguousBoundary
            } else {
                RejectionReason::InsufficientBoundary
            });
        }

        let base_quality = quality.clone();
        let candidate_context = CandidateEvaluationContext {
            candidates: &candidates,
            view,
            intrinsics: k,
            initial_center_image_px: initial,
            proposal_radius_px: radius,
            bbox_width_px: right - left,
            bbox_height_px: bottom - top,
            background_rgb: bg,
            sample: &sample,
            config: cfg,
            deadline: end,
            base_quality: &base_quality,
            detection_index: proposal.detection_index,
        };
        let mut attempts = Vec::with_capacity(representatives.len());
        let mut first_rejection: Option<(RejectionReason, BallLocalizationQuality)> = None;
        for (attempt_index, representative) in representatives.iter().enumerate() {
            deadline(Instant::now(), end)?;
            // Qualification must preserve the globally coherent path. Selecting
            // nearest transitions from all rays would undo the graph search and
            // can oscillate between mutually incompatible local edge families.
            let path_context = CandidateEvaluationContext {
                candidates: &hypothesis_paths[representative.hypothesis.fit_id],
                ..candidate_context
            };
            let mut attempt = evaluate_candidate(&representative.hypothesis, &path_context, trace);
            attempt.quality.candidate_representatives_evaluated = attempt_index + 1;
            attempt.quality.candidate_search_truncated = search_truncated;
            let attempt_deadline_exceeded =
                attempt.rejection == Some(RejectionReason::DeadlineExceeded);
            if let Some(reason) = attempt.rejection {
                if reason != RejectionReason::DeadlineExceeded && first_rejection.is_none() {
                    first_rejection = Some((reason, attempt.quality.clone()));
                }
            }
            if trace.is_some() && !attempt.diagnostic_outline_image_px.is_empty() {
                record_trace(trace, proposal.detection_index, &attempt.quality, || {
                    BallLocalizationAuditEventData::RefinedFit {
                        fit_id: attempt.fit_id,
                        center_local_px: attempt.fit.center,
                        center_image_px: attempt.center_image_px,
                        radius_local_px: attempt.fit.radius,
                        outline_image_px: attempt.diagnostic_outline_image_px.clone(),
                    }
                });
                if !attempt_deadline_exceeded {
                    deadline(Instant::now(), end)?;
                }
            }
            if trace.is_some() {
                record_trace(trace, proposal.detection_index, &attempt.quality, || {
                    BallLocalizationAuditEventData::CandidateEvaluation {
                        fit_id: attempt.fit_id,
                        center_image_px: attempt.center_image_px,
                        supported_rays: attempt.supported_rays,
                        residual_rms_px: attempt.quality.residual_rms_px,
                        source_residual_rms_px: attempt.source_residual_rms_px,
                        rejection: attempt.rejection,
                        quality: Box::new(attempt.quality.clone()),
                        outline_image_px: attempt.diagnostic_outline_image_px.clone(),
                    }
                });
                if !attempt_deadline_exceeded {
                    deadline(Instant::now(), end)?;
                }
            }
            if attempt_deadline_exceeded {
                *quality = attempt.quality;
                return Err(RejectionReason::DeadlineExceeded);
            }
            attempts.push(attempt);
        }

        let mut qualified_indices = attempts
            .iter()
            .enumerate()
            .filter_map(|(index, attempt)| attempt.is_qualified().then_some(index))
            .collect::<Vec<_>>();
        qualified_indices.sort_by(|&a, &b| candidate_attempt_rank(&attempts[a], &attempts[b]));
        if qualified_indices.is_empty() {
            if let Some((reason, mut failed_quality)) = first_rejection {
                failed_quality.candidate_representatives_evaluated = attempts.len();
                failed_quality.candidate_search_truncated = search_truncated;
                failed_quality.qualified_candidate_count = 0;
                *quality = failed_quality;
                if search_truncated {
                    quality.stage = BallFitStage::AmbiguityAssessment;
                    return Err(RejectionReason::AmbiguousBoundary);
                }
                return Err(reason);
            }
            quality.candidate_representatives_evaluated = attempts.len();
            quality.candidate_search_truncated = search_truncated;
            return Err(if search_truncated {
                quality.stage = BallFitStage::AmbiguityAssessment;
                RejectionReason::AmbiguousBoundary
            } else {
                RejectionReason::InsufficientBoundary
            });
        }

        let best_index = qualified_indices[0];
        let best = &attempts[best_index];
        let distinct_qualified_alternative =
            has_distinct_qualified_alternative(&attempts, &qualified_indices, cfg, end)?;
        if search_truncated || distinct_qualified_alternative {
            *quality = best.quality.clone();
            quality.candidate_representatives_evaluated = attempts.len();
            quality.qualified_candidate_count = qualified_indices.len();
            quality.candidate_search_truncated = search_truncated;
            quality.stage = BallFitStage::AmbiguityAssessment;
            return Err(RejectionReason::AmbiguousBoundary);
        }

        let best = &attempts[best_index];
        let center = best.center_image_px.expect("qualified candidate center");
        let outline = best.outline.clone().expect("qualified candidate outline");
        let change = distance(axis, center);
        axis = center;
        *quality = best.quality.clone();
        quality.candidate_representatives_evaluated = attempts.len();
        quality.qualified_candidate_count = qualified_indices.len();
        quality.candidate_search_truncated = false;
        final_fit = Some((center, outline));
        if change < cfg.convergence_px {
            break;
        }
    }
    deadline(Instant::now(), end)?;
    final_fit.ok_or(RejectionReason::NumericalFailure)
}

#[cfg(test)]
fn project_outline(
    k: &CameraIntrinsics,
    view: View,
    fit: Circle,
    center: [f64; 2],
    cfg: &BallLocalizationConfig,
    end: Instant,
    quality: &mut BallLocalizationQuality,
) -> Result<BallOutline, RejectionReason> {
    let points = source_outline_points(k, view, fit, cfg.outline_samples, end)?;
    outline_from_source_points(k, fit, center, &points, end, quality)
}

fn outline_from_source_points(
    k: &CameraIntrinsics,
    fit: Circle,
    center: [f64; 2],
    points: &[[f64; 2]],
    end: Instant,
    quality: &mut BallLocalizationQuality,
) -> Result<BallOutline, RejectionReason> {
    quality.stage = BallFitStage::OutlineProjection;
    deadline(Instant::now(), end)?;
    let mut outline_points = Vec::with_capacity(points.len());
    let mut extent = [0.0f64; 2];
    for &p in points {
        deadline(Instant::now(), end)?;
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
        outline_points.push(p);
    }
    quality.stage = BallFitStage::Complete;
    Ok(BallOutline {
        points_image_px: outline_points,
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
    fn cyclic_hypothesis_audit_preserves_complete_path_provenance() {
        let value = serde_json::to_value(BallLocalizationAuditEventData::CyclicHypothesis {
            fit_id: 4,
            path_cost: 2.5,
            candidate_indices: vec![11, 29, 47],
            center_local_px: [0.25, -0.5],
            center_image_px: Some([101., 202.]),
            radius_local_px: 12.5,
            support: 3,
            source_residual_rms_px: 0.2,
        })
        .unwrap();
        assert_eq!(BALL_LOCALIZATION_AUDIT_VERSION, 4);
        assert_eq!(value["kind"], "cyclic_hypothesis");
        assert_eq!(value["candidate_indices"], serde_json::json!([11, 29, 47]));
    }

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
    use super::{
        choose_diverse_representatives, has_distinct_qualified_alternative, irls,
        outline_disagreement_px, residual_is_plausible, same_source_boundary,
        source_residual_rms_px, strongest_distinct_ray_transitions, within_center_shift_budget,
        within_refined_radius_bounds, BallFitStage, BallLocalizationConfig,
        BallLocalizationQuality, BallOutline, CameraIntrinsics, Candidate, CandidateAttempt,
        Circle, RawHypothesis, RejectionReason, View,
    };
    use std::time::{Duration, Instant};

    fn source_camera() -> CameraIntrinsics {
        CameraIntrinsics {
            source_width_px: 480,
            source_height_px: 300,
            fx: 260.0,
            fy: 273.0,
            cx: 240.0,
            cy: 150.0,
        }
    }

    #[test]
    fn jittered_raw_hypotheses_collapse_to_one_source_boundary() {
        let k = source_camera();
        let axis = [240.0, 150.0];
        let view = View::new(&k, axis);
        let hypotheses = [
            RawHypothesis {
                fit_id: 0,
                fit: Circle {
                    center: [0.0, 0.0],
                    radius: 20.0,
                },
                supported_rays: 96,
                source_residual_rms_px: 0.2,
                center_image_px: axis,
            },
            RawHypothesis {
                fit_id: 1,
                fit: Circle {
                    center: [0.12, -0.08],
                    radius: 20.07,
                },
                supported_rays: 95,
                source_residual_rms_px: 0.21,
                center_image_px: view.source(&k, [0.12, -0.08]).unwrap(),
            },
            RawHypothesis {
                fit_id: 2,
                fit: Circle {
                    center: [-0.1, 0.06],
                    radius: 19.95,
                },
                supported_rays: 94,
                source_residual_rms_px: 0.22,
                center_image_px: view.source(&k, [-0.1, 0.06]).unwrap(),
            },
        ];
        let (representatives, truncated) = choose_diverse_representatives(
            &hypotheses,
            &k,
            view,
            &BallLocalizationConfig::default(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(representatives.len(), 1);
        assert!(!truncated);
    }

    #[test]
    fn distinct_fully_qualified_source_contours_are_ambiguous() {
        let first = qualified_attempt(0, [100.0, 100.0], 20.0, 96, 0.2);
        let second = qualified_attempt(1, [104.0, 100.0], 20.2, 95, 0.22);
        let attempts = [first, second];
        assert!(has_distinct_qualified_alternative(
            &attempts,
            &[0, 1],
            &BallLocalizationConfig::default(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap());
    }

    #[test]
    fn converged_candidates_with_matching_source_outlines_are_one_boundary() {
        let first = qualified_attempt(0, [100.0, 100.0], 20.0, 96, 0.2);
        let second = qualified_attempt(1, [100.2, 100.1], 20.15, 95, 0.22);
        let attempts = [first, second];
        assert_eq!(
            has_distinct_qualified_alternative(
                &attempts,
                &[0, 1],
                &BallLocalizationConfig::default(),
                Instant::now() + Duration::from_secs(1),
            ),
            Ok(false)
        );
    }

    #[test]
    fn alternative_residual_uses_normalized_rms_and_an_explicit_noise_floor() {
        let cfg = BallLocalizationConfig {
            maximum_alternative_residual_tolerance_px: 0.25,
            ..BallLocalizationConfig::default()
        };
        assert!(residual_is_plausible(95, 0.25, 96, 0.01, &cfg));
        assert!(!residual_is_plausible(95, 0.27, 96, 0.01, &cfg));
        assert!(!residual_is_plausible(90, 0.01, 96, 0.01, &cfg));
    }

    #[test]
    fn residual_rms_comparison_projects_measurements_to_source_pixels() {
        let k = source_camera();
        let fit = Circle {
            center: [0.0, 0.0],
            radius: 20.0,
        };
        let candidates = (0..96)
            .map(|ray| {
                let angle = std::f64::consts::TAU * ray as f64 / 96.0;
                Candidate {
                    point: [21.0 * angle.cos(), 21.0 * angle.sin()],
                    ray,
                    strength: 1.0,
                }
            })
            .collect::<Vec<_>>();
        let centered_view = View::new(&k, [k.cx, k.cy]);
        let off_axis_view = View::new(&k, [400.0, k.cy]);
        let end = Instant::now() + Duration::from_secs(1);
        let centered_rms =
            source_residual_rms_px(&candidates, fit, centered_view, &k, end).unwrap();
        let off_axis_rms =
            source_residual_rms_px(&candidates, fit, off_axis_view, &k, end).unwrap();
        assert!((centered_rms - 1.0).abs() < 1e-9);
        assert!((off_axis_rms - centered_rms).abs() > 0.01);
    }

    #[test]
    fn same_boundary_comparison_uses_source_image_pixels() {
        let k = source_camera();
        let view = View::new(&k, [60.0, 150.0]);
        let first_center = view.source(&k, [0.0, 0.0]).unwrap();
        let second_center = view.source(&k, [1.7, 0.0]).unwrap();
        let first_outline = circle_outline(first_center, 12.0);
        let second_outline = circle_outline(second_center, 12.0);
        assert!((first_center[0] - second_center[0]).abs() > 1.5);
        assert!(!same_source_boundary(
            first_center,
            &first_outline,
            second_center,
            &second_outline,
            &BallLocalizationConfig::default(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap());
    }

    #[test]
    fn same_center_with_materially_different_outlines_is_not_one_boundary() {
        let center = [80.0, 80.0];
        assert!(
            outline_disagreement_px(
                &circle_outline(center, 10.0),
                &circle_outline(center, 13.0),
                Instant::now() + Duration::from_secs(1),
            )
            .unwrap()
                > 1.5
        );
        assert!(!same_source_boundary(
            center,
            &circle_outline(center, 10.0),
            center,
            &circle_outline(center, 13.0),
            &BallLocalizationConfig::default(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap());
    }

    #[test]
    fn candidate_cap_records_plausible_unevaluated_boundary_and_deadline_is_checked() {
        let k = source_camera();
        let view = View::new(&k, [240.0, 150.0]);
        let hypotheses = [
            raw_hypothesis(0, [0.0, 0.0], 20.0, [240.0, 150.0]),
            raw_hypothesis(1, [5.0, 0.0], 20.0, [245.0, 150.0]),
        ];
        let cfg = BallLocalizationConfig {
            maximum_refined_candidates: 1,
            ..BallLocalizationConfig::default()
        };
        let (representatives, truncated) = choose_diverse_representatives(
            &hypotheses,
            &k,
            view,
            &cfg,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(representatives.len(), 1);
        assert!(truncated);

        let expired = Instant::now() - Duration::from_millis(1);
        assert!(matches!(
            choose_diverse_representatives(&hypotheses, &k, view, &cfg, expired),
            Err(RejectionReason::DeadlineExceeded)
        ));
    }

    #[test]
    fn deadline_during_qualified_alternative_search_is_not_ignored() {
        let attempts = [
            qualified_attempt(0, [100.0, 100.0], 20.0, 96, 0.2),
            qualified_attempt(1, [104.0, 100.0], 20.2, 95, 0.22),
        ];
        let expired = Instant::now() - Duration::from_millis(1);
        assert!(matches!(
            has_distinct_qualified_alternative(
                &attempts,
                &[0, 1],
                &BallLocalizationConfig::default(),
                expired,
            ),
            Err(RejectionReason::DeadlineExceeded)
        ));
    }

    #[test]
    fn source_shift_limit_is_fraction_of_the_shorter_box_side() {
        let max_shift = 0.2 * 20.0;
        assert!(within_center_shift_budget(max_shift, 20.0, 24.0, 0.2));
        assert!(!within_center_shift_budget(
            max_shift + 1e-9,
            20.0,
            24.0,
            0.2
        ));
    }

    #[test]
    fn adjacent_radial_samples_collapse_but_a_farther_transition_remains() {
        let selected = strongest_distinct_ray_transitions(
            vec![
                Candidate {
                    point: [10.0, 0.0],
                    ray: 0,
                    strength: 1.0,
                },
                Candidate {
                    point: [11.5, 0.0],
                    ray: 0,
                    strength: 0.9,
                },
                Candidate {
                    point: [14.5, 0.0],
                    ray: 0,
                    strength: 0.8,
                },
            ],
            3,
            2.0,
        );
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].point, [10.0, 0.0]);
        assert_eq!(selected[1].point, [14.5, 0.0]);
    }

    #[test]
    fn irls_radius_drift_outside_proposal_bounds_is_rejected() {
        let cfg = BallLocalizationConfig::default();
        let points = (0..96)
            .map(|ray| {
                let angle = std::f64::consts::TAU * ray as f64 / 96.0;
                [14.2 * angle.cos(), 14.2 * angle.sin()]
            })
            .collect::<Vec<_>>();
        let raw = Circle {
            center: [0.0, 0.0],
            radius: 14.0,
        };
        assert!(within_refined_radius_bounds(raw.radius, 10.0, &cfg));
        let refined = irls(raw, &points, &cfg, Instant::now() + Duration::from_secs(1)).unwrap();
        assert!(refined.radius > 14.0);
        assert!(!within_refined_radius_bounds(refined.radius, 10.0, &cfg));
    }

    fn raw_hypothesis(
        fit_id: usize,
        center: [f64; 2],
        radius: f64,
        source_center: [f64; 2],
    ) -> RawHypothesis {
        RawHypothesis {
            fit_id,
            fit: Circle { center, radius },
            supported_rays: 96 - fit_id,
            source_residual_rms_px: 0.2 + fit_id as f64 * 0.01,
            center_image_px: source_center,
        }
    }

    fn qualified_attempt(
        fit_id: usize,
        center: [f64; 2],
        radius: f64,
        supported_rays: usize,
        residual_rms_px: f64,
    ) -> CandidateAttempt {
        CandidateAttempt {
            fit_id,
            fit: Circle {
                center: [0.0, 0.0],
                radius,
            },
            supported_rays,
            source_residual_rms_px: Some(residual_rms_px),
            center_image_px: Some(center),
            outline: Some(BallOutline {
                points_image_px: circle_outline(center, radius),
                half_extent_image_px: [radius; 2],
                local_radius_px: radius,
            }),
            diagnostic_outline_image_px: Vec::new(),
            quality: BallLocalizationQuality {
                stage: BallFitStage::Complete,
                residual_rms_px: Some(residual_rms_px),
                ..BallLocalizationQuality::default()
            },
            rejection: None,
        }
    }

    fn circle_outline(center: [f64; 2], radius: f64) -> Vec<[f64; 2]> {
        (0..48)
            .map(|step| {
                let angle = std::f64::consts::TAU * step as f64 / 48.0;
                [
                    center[0] + radius * angle.cos(),
                    center[1] + radius * angle.sin(),
                ]
            })
            .collect()
    }
}
