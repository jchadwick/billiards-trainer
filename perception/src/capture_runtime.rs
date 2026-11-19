//! Rust-owned live capture lifecycle.
//!
//! This module composes the validated Rust capture-settings request, exclusive
//! source coordinator, RGB8 scene pipeline and request-bound v3 receiver
//! handshake. A frame receiver is withheld until source preflight and the v3
//! readiness exchange both succeed. Timeout and failure states retain the
//! coordinator so callers can explicitly stop and confirm source release.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;

use crate::calibration::{CalibrationContext, ValidatedTableCalibrationV1};
use crate::capture::{
    CaptureFrameReceiver, CaptureOwnerDisposition, CaptureOwnerStatus, CapturePreflightStatus,
    CapturePreflightWait, CaptureReceiveOutcome, CaptureRuntimeCoordinator, CaptureRuntimeWait,
    CaptureSourceFactory, CaptureWorkerReport, OwnedCaptureFrame,
};
use crate::capture_pipeline::{
    CapturePipelineConfig, CapturePipelineOutcome, CapturePipelineStats, CaptureScenePipeline,
    CaptureSkipReason, Rgb8FrameDetector,
};
use crate::capture_settings_wire::{parse_capture_settings_json, ResolvedCaptureSettings};
use crate::cue_geometry::{
    orient_cue_geometry_from_playing_surface, CueGeometry, CueGeometryConfig, CueProposal,
    SignedCueDirection,
};
use crate::cue_worker::{
    CueGeometryWorker, CueGeometryWorkerShutdown, CueGeometryWorkerState, CueWorkTag,
};
use crate::detector_worker::{
    DetectorWorkTag, DetectorWorker, DetectorWorkerShutdown, DetectorWorkerState,
};
use crate::frame_preview::LatestCapturedFrameSlot;
use crate::live_audit::LiveAuditRecorder;
use crate::motion_monitor::{
    MotionAssessment, MotionClassification, MotionMonitor, MotionMonitorConfig, MotionReason,
};
use crate::phase_scheduler::{
    CueEvidence, ImagePoint, LayoutObservation, MotionEvidence, Phase, PhaseScheduler,
    SchedulerConfig, SchedulerEvidence, TrustReason, WorkAvailability,
};
use crate::producer_diagnostics::ReporterHandle;
use crate::producer_diagnostics_contract::{ScheduledJobKind, ScheduledWorkState};
use crate::public_state_lifecycle::{
    CueGeometryStateV1, PerceptionStateV1, TableProjectionV1, TableVisibilityStateV1,
    TableVisibilityStatusV1,
};
use crate::scene_transport::SceneTransportStats;
use crate::table_visibility::{
    assess_table_visibility, Rgb8Frame as VisibilityRgb8Frame, TableVisibilityAssessment,
    TableVisibilityConfig, TableVisibilityReason, TableVisibilityReference, TableVisibilityStatus,
};
use crate::tracking::TrackingConfigV1;

/// Observable lifecycle for a live source session.
///
/// `Pending` includes readiness timeouts; it does not imply that ownership has
/// been released. `Failed` likewise keeps the coordinator alive until callers
/// invoke [`LiveCaptureRuntime::stop`] and observe its result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveCaptureLifecycle {
    Pending { session_id: String },
    Ready { session_id: String },
    Failed { session_id: String, reason: String },
    Stopping { session_id: String },
    Stopped { session_id: String },
}

/// A composed live camera-to-scene session with explicit source ownership.
///
/// Construction accepts the exact UTF-8 request bytes produced by the Rust
/// configuration authority.
/// The request is parsed once and used for source preflight, request digest,
/// frame limits and pipeline consistency checks. The pipeline is constructed
/// before the source factory can open hardware. Only `wait_until_ready` can
/// expose the sole receiver, and it does so only after the compatible report
/// has completed its v3 readiness handshake.
#[must_use = "retain the runtime through readiness and explicitly stop it to confirm source release"]
pub struct LiveCaptureRuntime<D: Rgb8FrameDetector> {
    session_id: String,
    settings: ResolvedCaptureSettings,
    coordinator: CaptureRuntimeCoordinator,
    pipeline: CaptureScenePipeline<D>,
    detector_worker: Option<DetectorWorker>,
    cue_worker: Option<CueGeometryWorker>,
    phase_scheduler: Option<PhaseScheduler>,
    phase_schedule_config: Option<SchedulerConfig>,
    motion_monitor: Option<MotionMonitor>,
    /// Read once at runtime construction so normal frame processing only pays
    /// one predictable branch and never formats or serializes diagnostics.
    motion_diagnostics_enabled: bool,
    cue_config: CueGeometryConfig,
    cue_proposals: Vec<CueProposal>,
    cue_geometry: Option<CueGeometry>,
    cue_observed_at: Option<Duration>,
    cue_observation_frame_id: Option<String>,
    session_clock_origin: Instant,
    last_scheduler_tick: Option<Duration>,
    phase_stats: PhaseRuntimeStats,
    diagnostics: Option<ReporterHandle>,
    scheduled_work_state: ScheduledWorkState,
    active_detector_started_at: Option<Instant>,
    active_cue_started_at: Option<Instant>,
    active_motion_started_at: Option<Instant>,
    scheduler_jobs_expected: u64,
    scheduler_jobs_completed: u64,
    scheduler_jobs_failed: u64,
    last_scheduled_completion_at: Option<Instant>,
    last_scheduled_failure_code: Option<&'static str>,
    last_scheduled_failure_at: Option<Instant>,
    recovery_started_at: Option<Duration>,
    table_visibility_config: Option<TableVisibilityConfig>,
    table_visibility_reference: Option<TableVisibilityReference>,
    latest_visibility: Option<(String, TableVisibilityAssessment)>,
    latest_visibility_observed_at: Option<Duration>,
    latest_visibility_capture_at: Option<Duration>,
    /// Capture-time high-water marks keep newer uncertain/contradictory
    /// appearance evidence from being overwritten by a later assessment of a
    /// different frame while an older detector job is still in flight.
    latest_unconfirmed_visibility_capture_at: Option<Duration>,
    latest_not_visible_capture_at: Option<Duration>,
    decision_frame_id: Option<String>,
    decision_frame_dimensions: Option<(u32, u32)>,
    /// Detector frame backing the currently trusted whole-table layout. It is
    /// carried separately from later phase decisions so the gateway can fail
    /// closed if this clean scene expires during transport recovery.
    layout_basis_frame_id: Option<String>,
    receiver: Option<CaptureFrameReceiver>,
    lifecycle: LiveCaptureLifecycle,
}

impl<D: Rgb8FrameDetector> std::fmt::Debug for LiveCaptureRuntime<D> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveCaptureRuntime")
            .field("session_id", &self.session_id)
            .field(
                "configuration_revision",
                &self.settings.configuration_revision,
            )
            .field("request_sha256", &self.settings.request_sha256)
            .field("coordinator", &self.coordinator)
            .field("lifecycle", &self.lifecycle)
            .field("receiver_claimed", &self.receiver.is_some())
            .field("phase_scheduler_enabled", &self.phase_scheduler.is_some())
            .field("detector_worker", &self.detector_worker)
            .finish_non_exhaustive()
    }
}

impl<D: Rgb8FrameDetector + Send + 'static> LiveCaptureRuntime<D> {
    /// Parse the exact request, validate duplicated pipeline settings, create
    /// the pipeline, derive coordinator budgets, and start the injected source.
    ///
    /// A successful return is `Pending`; source open and compatibility checks
    /// complete asynchronously. This method intentionally does not wait for
    /// readiness, so callers can retain the runtime across bounded waits.
    pub fn start<F: CaptureSourceFactory>(
        exact_request_json: &str,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        source_factory: F,
    ) -> Result<Self> {
        Self::start_with_identity(
            exact_request_json,
            None,
            detector,
            tracking,
            pipeline_config,
            None,
            None,
            None,
            None,
            None,
            source_factory,
        )
    }

    /// Start a live source with a caller-owned identity shared by the process
    /// protocol and every emitted scene.
    ///
    /// The gateway uses this when it creates the producer session identity
    /// before spawning Rust. The source report, v3 acknowledgement and scene
    /// snapshots then carry exactly the same value instead of a second,
    /// process-generated identity.
    pub fn start_with_session_id<F: CaptureSourceFactory>(
        exact_request_json: &str,
        session_id: impl Into<String>,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        source_factory: F,
    ) -> Result<Self> {
        Self::start_with_identity(
            exact_request_json,
            Some(session_id.into()),
            detector,
            tracking,
            pipeline_config,
            None,
            None,
            None,
            None,
            None,
            source_factory,
        )
    }

    /// Start a live source with an optional performance-audit recorder. Audit
    /// export uses a separate bounded loopback queue and never changes scene
    /// payloads or blocks source-frame processing.
    pub fn start_with_session_id_and_audit<F: CaptureSourceFactory>(
        exact_request_json: &str,
        session_id: impl Into<String>,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        audit: Option<LiveAuditRecorder>,
        source_factory: F,
    ) -> Result<Self> {
        Self::start_with_identity(
            exact_request_json,
            Some(session_id.into()),
            detector,
            tracking,
            pipeline_config,
            audit,
            None,
            None,
            None,
            None,
            source_factory,
        )
    }

    /// Start the normal application path with explicit table-visibility
    /// evidence. Keeping this separate from the compatibility constructors
    /// makes legacy replay callers opt in while live guidance always fails
    /// closed when a reference is absent.
    // These independent inputs retain the same explicit ownership as the
    // neighboring audit and frame-preview constructors.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_session_id_audit_and_visibility<F: CaptureSourceFactory>(
        exact_request_json: &str,
        session_id: impl Into<String>,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        audit: Option<LiveAuditRecorder>,
        table_visibility_config: TableVisibilityConfig,
        table_visibility_reference: Option<TableVisibilityReference>,
        source_factory: F,
    ) -> Result<Self> {
        Self::start_with_identity(
            exact_request_json,
            Some(session_id.into()),
            detector,
            tracking,
            pipeline_config,
            audit,
            None,
            None,
            Some(table_visibility_config),
            table_visibility_reference,
            source_factory,
        )
    }

    /// Start a live capture session with a latest-captured-frame preview slot.
    /// The slot is populated by the capture publisher before the inference
    /// receiver, and is bound only after source preflight accepts the request.
    // These are independent runtime inputs with their existing public ownership.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_session_id_audit_and_frame_preview<F: CaptureSourceFactory>(
        exact_request_json: &str,
        session_id: impl Into<String>,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        audit: Option<LiveAuditRecorder>,
        frame_preview: Arc<LatestCapturedFrameSlot>,
        source_factory: F,
    ) -> Result<Self> {
        Self::start_with_identity(
            exact_request_json,
            Some(session_id.into()),
            detector,
            tracking,
            pipeline_config,
            audit,
            Some(frame_preview),
            None,
            None,
            None,
            source_factory,
        )
    }

    /// Start a preview-enabled application path with calibrated-table
    /// visibility assessment active in the capture owner.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_session_id_audit_frame_preview_and_visibility<F: CaptureSourceFactory>(
        exact_request_json: &str,
        session_id: impl Into<String>,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        audit: Option<LiveAuditRecorder>,
        frame_preview: Arc<LatestCapturedFrameSlot>,
        table_visibility_config: TableVisibilityConfig,
        table_visibility_reference: Option<TableVisibilityReference>,
        source_factory: F,
    ) -> Result<Self> {
        Self::start_with_identity(
            exact_request_json,
            Some(session_id.into()),
            detector,
            tracking,
            pipeline_config,
            audit,
            Some(frame_preview),
            None,
            Some(table_visibility_config),
            table_visibility_reference,
            source_factory,
        )
    }

    /// Start with a Rust-owned durable calibration snapshot. The exact request
    /// JSON is still parsed and hashed unchanged; only its in-memory effective
    /// calibration state is replaced before source preflight and pipeline
    /// validation.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_session_id_audit_frame_preview_and_calibration<F: CaptureSourceFactory>(
        exact_request_json: &str,
        session_id: impl Into<String>,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        audit: Option<LiveAuditRecorder>,
        frame_preview: Arc<LatestCapturedFrameSlot>,
        calibration_revision: u64,
        calibration: Option<ValidatedTableCalibrationV1>,
        table_visibility_config: TableVisibilityConfig,
        table_visibility_reference: Option<TableVisibilityReference>,
        source_factory: F,
    ) -> Result<Self> {
        Self::start_with_identity(
            exact_request_json,
            Some(session_id.into()),
            detector,
            tracking,
            pipeline_config,
            audit,
            Some(frame_preview),
            Some((calibration_revision, calibration)),
            Some(table_visibility_config),
            table_visibility_reference,
            source_factory,
        )
    }

    // Keep audit and preview independently optional while sharing startup logic.
    #[allow(clippy::too_many_arguments)]
    fn start_with_identity<F: CaptureSourceFactory>(
        exact_request_json: &str,
        requested_session_id: Option<String>,
        detector: D,
        tracking: TrackingConfigV1,
        pipeline_config: CapturePipelineConfig,
        audit: Option<LiveAuditRecorder>,
        frame_preview: Option<Arc<LatestCapturedFrameSlot>>,
        calibration_store_state: Option<(u64, Option<ValidatedTableCalibrationV1>)>,
        table_visibility_config: Option<TableVisibilityConfig>,
        table_visibility_reference: Option<TableVisibilityReference>,
        source_factory: F,
    ) -> Result<Self> {
        let scheduler_config = pipeline_config.phase_schedule.clone();
        let session_clock_origin = Instant::now();
        let motion_diagnostics_enabled =
            std::env::var_os("BILLIARDS_MOTION_DIAGNOSTICS").is_some_and(|value| value == "1");
        let mut settings = parse_capture_settings_json(exact_request_json)
            .context("invalid exact capture-settings request")?;
        if let Some(config) = table_visibility_config.as_ref() {
            config
                .validate()
                .context("invalid calibrated-table visibility configuration")?;
        }
        if let Some((revision, calibration)) = calibration_store_state {
            settings
                .expected
                .apply_calibration_store_state(revision, calibration)
                .context("Rust-owned calibration store state does not match the capture context")?;
        }

        // Establish the pipeline clock origin and its bounded scene writer
        // before the capture worker is permitted to open its source.
        let mut pipeline =
            CaptureScenePipeline::new_with_audit(detector, tracking, pipeline_config, audit)
                .context("failed to create live capture-to-scene pipeline")?;
        pipeline
            .validate_expected_capture_settings(&settings.expected)
            .map_err(|error| {
                anyhow!("pipeline configuration disagrees with capture request: {error}")
            })?;

        // Phase-aware live sessions move the mutable model into one dedicated
        // worker. Replay and compatibility callers without a schedule retain
        // the existing synchronous pipeline.
        let phase_schedule_config = scheduler_config.clone();
        let (detector_worker, phase_scheduler, motion_monitor) =
            if let Some(config) = scheduler_config {
                let scheduler = PhaseScheduler::new(config)
                    .context("invalid phase-aware perception schedule")?;
                let motion = MotionMonitor::new(MotionMonitorConfig::default())
                    .context("invalid motion monitor configuration")?;
                let model = pipeline
                    .take_detector_for_worker()
                    .context("failed to transfer detector model to live worker")?;
                let worker = DetectorWorker::spawn(model)
                    .context("failed to create dedicated detector worker")?;
                (Some(worker), Some(scheduler), Some(motion))
            } else {
                (None, None, None)
            };
        let cue_config = CueGeometryConfig::default();
        let cue_worker = phase_schedule_config
            .as_ref()
            .is_some_and(|config| {
                [
                    config.recovering.cue_hz,
                    config.pre_shot.cue_hz,
                    config.aiming.cue_hz,
                    config.shot.cue_hz,
                ]
                .iter()
                .any(|rate| *rate > 0.0)
            })
            .then(|| CueGeometryWorker::spawn(cue_config))
            .transpose()
            .context("failed to create dedicated cue-geometry worker")?;

        // The request, rather than a second caller-supplied configuration,
        // owns both per-frame and aggregate application-buffer limits.
        let mut coordinator = CaptureRuntimeCoordinator::new(
            settings.expected.frame_limit(),
            settings.expected.aggregate_frame_limit(),
        )
        .context("failed to configure live capture frame budgets")?;
        let session_id = match (requested_session_id, frame_preview) {
            (Some(session_id), Some(preview)) => coordinator
                .start_preflighted_with_session_id_and_frame_preview(
                    settings.expected.clone(),
                    session_id,
                    preview,
                    source_factory,
                ),
            (Some(session_id), None) => coordinator.start_preflighted_with_session_id(
                settings.expected.clone(),
                session_id,
                source_factory,
            ),
            (None, Some(_)) => {
                bail!("frame preview requires an explicit capture session identity")
            }
            (None, None) => {
                coordinator.start_preflighted(settings.expected.clone(), source_factory)
            }
        }
        .context("failed to start the live capture source")?;

        Ok(Self {
            lifecycle: LiveCaptureLifecycle::Pending {
                session_id: session_id.clone(),
            },
            session_id,
            settings,
            coordinator,
            pipeline,
            detector_worker,
            cue_worker,
            phase_scheduler,
            phase_schedule_config,
            motion_monitor,
            motion_diagnostics_enabled,
            cue_config,
            cue_proposals: Vec::new(),
            cue_geometry: None,
            cue_observed_at: None,
            cue_observation_frame_id: None,
            session_clock_origin,
            last_scheduler_tick: None,
            phase_stats: PhaseRuntimeStats::default(),
            diagnostics: None,
            scheduled_work_state: ScheduledWorkState::Unknown,
            active_detector_started_at: None,
            active_cue_started_at: None,
            active_motion_started_at: None,
            scheduler_jobs_expected: 0,
            scheduler_jobs_completed: 0,
            scheduler_jobs_failed: 0,
            last_scheduled_completion_at: None,
            last_scheduled_failure_code: None,
            last_scheduled_failure_at: None,
            recovery_started_at: None,
            table_visibility_config,
            table_visibility_reference,
            latest_visibility: None,
            latest_visibility_observed_at: None,
            latest_visibility_capture_at: None,
            latest_unconfirmed_visibility_capture_at: None,
            latest_not_visible_capture_at: None,
            decision_frame_id: None,
            decision_frame_dimensions: None,
            layout_basis_frame_id: None,
            receiver: None,
        })
    }

    /// Current readiness state. A timeout leaves this at `Pending` and keeps
    /// the source coordinator and its owner gate in this runtime.
    pub fn lifecycle(&self) -> &LiveCaptureLifecycle {
        &self.lifecycle
    }

    /// Attach the independent diagnostics reporter before readiness begins.
    /// It never changes capture behavior and only receives compact stage data.
    pub fn attach_diagnostics(&mut self, diagnostics: ReporterHandle) {
        self.diagnostics = Some(diagnostics);
        self.publish_scheduled_work_evidence();
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn configuration_revision(&self) -> u64 {
        self.settings.configuration_revision
    }

    pub fn request_sha256(&self) -> &str {
        &self.settings.request_sha256
    }

    /// Wait for source preflight and v3 readiness, bounded by the capture
    /// preflight timeout plus the scene transport's configured handshake
    /// deadlines. The receiver is claimed only after both checks succeed.
    pub fn wait_until_ready(&mut self, timeout: Duration) -> &LiveCaptureLifecycle {
        if !matches!(self.lifecycle, LiveCaptureLifecycle::Pending { .. }) {
            return &self.lifecycle;
        }
        let Some(wait) = self.coordinator.wait_for_preflight(timeout) else {
            self.fail("capture coordinator has no active preflight state".to_owned());
            return &self.lifecycle;
        };
        let status = match wait {
            CapturePreflightWait::Ready(status) | CapturePreflightWait::TimedOut(status) => status,
        };

        match status {
            CapturePreflightStatus::Pending { .. } => {}
            CapturePreflightStatus::Compatible(compatibility) => {
                if let Err(error) = self.pipeline.bind_preflight(*compatibility) {
                    self.fail(format!("live scene readiness failed: {error}"));
                    return &self.lifecycle;
                }
                match self.coordinator.claim_receiver() {
                    Ok(receiver) => {
                        self.receiver = Some(receiver);
                        self.lifecycle = LiveCaptureLifecycle::Ready {
                            session_id: self.session_id.clone(),
                        };
                    }
                    Err(error) => {
                        self.fail(format!("failed to claim sole frame receiver: {error}"))
                    }
                }
            }
            CapturePreflightStatus::Rejected { error, .. } => {
                self.fail(format!("capture source preflight rejected: {error}"));
            }
            CapturePreflightStatus::Failed { reason, .. } => {
                self.fail(format!("capture source preflight failed: {reason}"));
            }
            CapturePreflightStatus::Stopped { .. } => {
                self.fail("capture source stopped before readiness".to_owned());
            }
        }
        &self.lifecycle
    }

    /// Process at most one source observation through the real RGB8 pipeline.
    /// Calling this before v3 readiness is an error and cannot consume a frame.
    pub fn receive_one(
        &mut self,
        timeout: Duration,
    ) -> Result<CaptureReceiveOutcome<Result<CapturePipelineOutcome>>> {
        if !matches!(self.lifecycle, LiveCaptureLifecycle::Ready { .. }) {
            return Err(anyhow!(
                "live capture receiver is unavailable while lifecycle is {:?}",
                self.lifecycle
            ));
        }
        let outcome = if self.phase_scheduler.is_some() {
            self.receive_phase_aware(timeout)?
        } else {
            let receiver = self
                .receiver
                .as_mut()
                .context("ready runtime does not own its frame receiver")?;
            self.pipeline.receive_one(receiver, timeout)
        };
        match &outcome {
            CaptureReceiveOutcome::Frame(Err(error)) => {
                self.fail(format!("live frame processing failed: {error:#}"));
            }
            CaptureReceiveOutcome::ProducerFailed(reason) => {
                self.invalidate_phase_context();
                self.fail(format!("live capture source failed: {reason}"));
            }
            CaptureReceiveOutcome::Cancelled => {
                self.invalidate_phase_context();
                self.fail(
                    "live capture source was cancelled; explicit stop is required".to_owned(),
                );
            }
            CaptureReceiveOutcome::Closed => {
                self.lifecycle = LiveCaptureLifecycle::Stopping {
                    session_id: self.session_id.clone(),
                };
            }
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
        }
        Ok(outcome)
    }

    /// Explicitly interrupt and release the source. `TimedOut` preserves the
    /// coordinator and owner lease in this runtime, so the caller can retry
    /// stop or continue observing the same session without opening a second.
    pub fn stop(&mut self, timeout: Duration) -> CaptureRuntimeWait {
        if matches!(self.lifecycle, LiveCaptureLifecycle::Stopped { .. }) {
            return CaptureRuntimeWait::Idle;
        }
        self.receiver = None;
        self.lifecycle = LiveCaptureLifecycle::Stopping {
            session_id: self.session_id.clone(),
        };
        let wait = self.coordinator.stop(timeout);
        match &wait {
            CaptureRuntimeWait::TimedOut => {
                self.lifecycle = LiveCaptureLifecycle::Failed {
                    session_id: self.session_id.clone(),
                    reason: "source release is still pending; retain this runtime and retry stop"
                        .to_owned(),
                };
            }
            CaptureRuntimeWait::Completed(report) => self.apply_stop_report(report),
            CaptureRuntimeWait::Idle => {
                if matches!(
                    self.coordinator.owner_status(),
                    CaptureOwnerStatus::Available
                ) {
                    self.pipeline.shutdown();
                    self.lifecycle = LiveCaptureLifecycle::Stopped {
                        session_id: self.session_id.clone(),
                    };
                } else {
                    self.fail(format!(
                        "capture coordinator is idle but source ownership is {:?}",
                        self.coordinator.owner_status()
                    ));
                }
            }
        }
        if self.detector_worker.as_mut().is_some_and(|worker| {
            worker.shutdown(timeout) == DetectorWorkerShutdown::TimedOutDetached
        }) {
            self.fail("detector worker did not stop before its shutdown bound; the process supervisor must contain the stuck native inference".to_owned());
        }
        if self.cue_worker.as_mut().is_some_and(|worker| {
            worker.shutdown(timeout) == CueGeometryWorkerShutdown::TimedOutDetached
        }) {
            self.fail("cue-geometry worker did not stop before its shutdown bound; the process supervisor must contain the stuck worker".to_owned());
        }
        wait
    }

    /// Observe the source worker without relinquishing its ownership handle.
    pub fn wait_for_source_exit(&mut self, timeout: Duration) -> CaptureRuntimeWait {
        self.coordinator.wait_timeout(timeout)
    }

    pub fn owner_status(&self) -> CaptureOwnerStatus {
        self.coordinator.owner_status()
    }

    fn note_scheduler_decision(&mut self, decision: crate::phase_scheduler::ScheduleDecision) {
        if decision.run_motion || decision.run_ball || decision.run_cue {
            self.scheduled_work_state = ScheduledWorkState::Due;
        } else {
            self.scheduled_work_state =
                if scheduler_phase_is_gated(self.phase_schedule_config.as_ref(), decision.phase) {
                    ScheduledWorkState::PhaseGated
                } else {
                    ScheduledWorkState::Idle
                };
        }
        self.publish_scheduled_work_evidence();
    }

    fn start_scheduled_job(&mut self, kind: ScheduledJobKind) {
        let started_at = Instant::now();
        self.scheduler_jobs_expected = self.scheduler_jobs_expected.saturating_add(1);
        match kind {
            ScheduledJobKind::MotionAssessment => self.active_motion_started_at = Some(started_at),
            ScheduledJobKind::BallDetection => self.active_detector_started_at = Some(started_at),
            ScheduledJobKind::CueGeometry => self.active_cue_started_at = Some(started_at),
        }
        self.scheduled_work_state = ScheduledWorkState::InFlight;
        self.publish_scheduled_work_evidence();
    }

    fn finish_scheduled_job(&mut self, kind: ScheduledJobKind, failure_code: Option<&'static str>) {
        match kind {
            ScheduledJobKind::MotionAssessment => self.active_motion_started_at = None,
            ScheduledJobKind::BallDetection => self.active_detector_started_at = None,
            ScheduledJobKind::CueGeometry => self.active_cue_started_at = None,
        }
        let now = Instant::now();
        if let Some(code) = failure_code {
            self.scheduler_jobs_failed = self.scheduler_jobs_failed.saturating_add(1);
            self.last_scheduled_failure_code = Some(code);
            self.last_scheduled_failure_at = Some(now);
            self.scheduled_work_state = ScheduledWorkState::Failed;
        } else {
            self.scheduler_jobs_completed = self.scheduler_jobs_completed.saturating_add(1);
            self.last_scheduled_completion_at = Some(now);
            self.scheduled_work_state = ScheduledWorkState::Idle;
        }
        self.publish_scheduled_work_evidence();
    }

    fn publish_scheduled_work_evidence(&mut self) {
        let Some(diagnostics) = self.diagnostics.as_ref() else {
            return;
        };
        let detector_state = self.detector_worker.as_ref().map(DetectorWorker::state);
        let cue_state = self.cue_worker.as_ref().map(CueGeometryWorker::state);
        let detector_active = detector_state == Some(DetectorWorkerState::InferenceInFlight);
        let cue_active = cue_state == Some(CueGeometryWorkerState::EstimationInFlight);
        let mut active = Vec::with_capacity(3);
        if let Some(started_at) = self.active_motion_started_at {
            active.push((Some(started_at), ScheduledJobKind::MotionAssessment));
        }
        if detector_active {
            active.push((
                self.active_detector_started_at,
                ScheduledJobKind::BallDetection,
            ));
        }
        if cue_active {
            active.push((self.active_cue_started_at, ScheduledJobKind::CueGeometry));
        }
        let active_without_start = active.iter().find(|(started_at, _)| started_at.is_none());
        let earliest_active = active
            .iter()
            .filter_map(|(started_at, kind)| started_at.map(|started_at| (started_at, *kind)))
            .min_by_key(|(started_at, _)| *started_at);
        let (started_at, active_job_kind) = if let Some((_, kind)) = active_without_start {
            (None, Some(*kind))
        } else {
            earliest_active
                .map(|(started_at, kind)| (Some(started_at), Some(kind)))
                .unwrap_or((None, None))
        };
        let in_flight = !active.is_empty();
        let state = if in_flight {
            ScheduledWorkState::InFlight
        } else if detector_state == Some(DetectorWorkerState::ResultReady)
            || cue_state == Some(CueGeometryWorkerState::ResultReady)
        {
            ScheduledWorkState::ResultReady
        } else if detector_state == Some(DetectorWorkerState::Failed)
            || cue_state == Some(CueGeometryWorkerState::Failed)
        {
            ScheduledWorkState::Failed
        } else {
            self.scheduled_work_state
        };
        diagnostics.set_scheduled_work(crate::producer_diagnostics::ScheduledWorkEvidence {
            scheduler_state: state,
            in_flight,
            active_job_kind,
            work_started_at: started_at,
            expected_jobs: self.scheduler_jobs_expected,
            completed_jobs: self.scheduler_jobs_completed,
            failed_jobs: self.scheduler_jobs_failed,
            last_completion_at: self.last_scheduled_completion_at,
            last_failure_code: self.last_scheduled_failure_code,
            last_failure_at: self.last_scheduled_failure_at,
        });
    }

    pub fn pipeline_stats(&self) -> CapturePipelineStats {
        self.pipeline.stats()
    }

    /// Counters for actual phase-aware work, skips, and retained-layout time.
    /// Callers can combine these with elapsed session time to calculate rates.
    pub fn phase_stats(&self) -> PhaseRuntimeStats {
        self.phase_stats
    }

    /// Complete measurement-window diagnostics for rate qualification. The
    /// rates are derived from elapsed session time, never requested settings.
    pub fn phase_diagnostics(&self) -> Option<PhaseRuntimeDiagnostics> {
        self.phase_scheduler.as_ref()?;
        let elapsed = self.session_elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let rate = |count: u64| (elapsed_seconds > 0.0).then_some(count as f64 / elapsed_seconds);
        Some(PhaseRuntimeDiagnostics {
            elapsed_ms: millis(elapsed),
            captured_fps: rate(self.phase_stats.captured_frames),
            motion_checks_per_second: rate(self.phase_stats.motion_checks),
            detector_calls_per_second: rate(self.phase_stats.detector_calls),
            cue_observations_per_second: rate(self.phase_stats.cue_observations),
            work: self.phase_stats,
        })
    }

    /// Current phase and evidence validity, if phase scheduling is enabled.
    pub fn scheduler_evidence(&self) -> Result<Option<SchedulerEvidence>> {
        let now = self.session_elapsed();
        self.phase_scheduler
            .as_ref()
            .map(|scheduler| scheduler.evidence(now).map_err(anyhow::Error::from))
            .transpose()
    }

    /// Latest independent cue-line estimate with age measured from its own
    /// image observation, never from the ball-layout timestamp.
    pub fn cue_geometry(&self) -> Option<CueGeometry> {
        let mut cue = self.cue_geometry?;
        cue.observation_age_ms = self
            .cue_observed_at
            .map(|at| {
                self.session_elapsed()
                    .saturating_sub(at)
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64
            })
            .unwrap_or_default();
        Some(cue)
    }

    pub fn transport_stats(&self) -> SceneTransportStats {
        self.pipeline.transport_stats()
    }

    fn apply_stop_report(&mut self, report: &CaptureWorkerReport) {
        self.pipeline.shutdown();
        match &report.owner_disposition {
            CaptureOwnerDisposition::Released => {
                self.lifecycle = LiveCaptureLifecycle::Stopped {
                    session_id: self.session_id.clone(),
                };
            }
            CaptureOwnerDisposition::Quarantined { reason } => {
                self.fail(format!("source release could not be confirmed: {reason}"));
            }
        }
    }

    fn fail(&mut self, reason: String) {
        self.lifecycle = LiveCaptureLifecycle::Failed {
            session_id: self.session_id.clone(),
            reason,
        };
    }

    /// Keep source consumption and inexpensive checks moving while the model
    /// runs in its own thread. Each newest frame's motion invalidation runs
    /// before accepting a completion, which is then consumed before dispatch.
    fn receive_phase_aware(
        &mut self,
        timeout: Duration,
    ) -> Result<CaptureReceiveOutcome<Result<CapturePipelineOutcome>>> {
        let received = self
            .receiver
            .as_mut()
            .context("ready runtime does not own its frame receiver")?
            .receive_owned_with_timeout(timeout);
        match received {
            CaptureReceiveOutcome::Frame(frame) => {
                self.phase_stats.captured_frames =
                    self.phase_stats.captured_frames.saturating_add(1);
                let result = self.observe_and_schedule_frame(frame);
                let (no_dispatch, completion) = match result {
                    Ok(result) => result,
                    Err(error) => {
                        if self.active_motion_started_at.is_some() {
                            self.finish_scheduled_job(
                                ScheduledJobKind::MotionAssessment,
                                Some("motion_assessment_failed"),
                            );
                        }
                        return Err(error);
                    }
                };
                if let Some(outcome) = completion {
                    return Ok(CaptureReceiveOutcome::Frame(Ok(outcome)));
                }
                if no_dispatch == CaptureSkipReason::DetectorOverloaded {
                    self.phase_stats.overload_skips =
                        self.phase_stats.overload_skips.saturating_add(1);
                } else {
                    self.phase_stats.intentional_skips =
                        self.phase_stats.intentional_skips.saturating_add(1);
                }
                Ok(CaptureReceiveOutcome::Frame(Ok(
                    CapturePipelineOutcome::Skipped(no_dispatch),
                )))
            }
            CaptureReceiveOutcome::TimedOut => {
                self.tick_expiries()?;
                if self.poll_cue_completion()? {
                    self.publish_perception_status(self.session_elapsed())?;
                }
                if let Some(outcome) = self.poll_detector_completion()? {
                    Ok(CaptureReceiveOutcome::Frame(Ok(outcome)))
                } else {
                    Ok(CaptureReceiveOutcome::TimedOut)
                }
            }
            CaptureReceiveOutcome::Closed => Ok(CaptureReceiveOutcome::Closed),
            CaptureReceiveOutcome::Cancelled => Ok(CaptureReceiveOutcome::Cancelled),
            CaptureReceiveOutcome::ProducerFailed(reason) => {
                self.invalidate_phase_context();
                Ok(CaptureReceiveOutcome::ProducerFailed(reason))
            }
        }
    }

    fn observe_and_schedule_frame(
        &mut self,
        mut frame: OwnedCaptureFrame,
    ) -> Result<(CaptureSkipReason, Option<CapturePipelineOutcome>)> {
        let now = self.session_elapsed();
        let scheduler = self
            .phase_scheduler
            .as_mut()
            .context("phase-aware runtime is missing its scheduler")?;
        let phase_before = scheduler.phase();
        // Scheduling happens exactly once for each source frame. Motion and
        // asynchronous cue observations may change the phase below; the
        // scheduler records those transitions and resets deadlines itself.
        // Refresh evidence after those observations, but do not tick again:
        // a second tick can consume a just-due slot that this frame cannot run.
        let decision = scheduler
            .tick(now, WorkAvailability::default())
            .context("phase scheduler rejected runtime clock")?;
        self.note_scheduler_decision(decision);
        self.accumulate_trusted_time(now);
        let motion_check_ran = decision.run_motion;

        if decision.run_motion {
            let mut visibility_requires_stale_layout = false;
            self.start_scheduled_job(ScheduledJobKind::MotionAssessment);
            let assessment = self
                .motion_monitor
                .as_mut()
                .context("phase-aware runtime is missing its motion monitor")?
                .observe_rgb8(
                    now,
                    frame.source().width_px,
                    frame.source().height_px,
                    frame.rgb8(),
                )
                .context("lightweight motion assessment failed")?;
            self.phase_stats.motion_checks = self.phase_stats.motion_checks.saturating_add(1);
            let evidence = match assessment.classification {
                MotionClassification::Quiet => MotionEvidence::Quiet,
                MotionClassification::Moving => MotionEvidence::Moving,
                MotionClassification::OccludedOrUncertain => MotionEvidence::OccludedOrUncertain,
            };
            // A stored appearance reference can provide an explicit known
            // occlusion signal, but its absence is not evidence of occlusion
            // and must not block calibrated geometry plus current motion data.
            if let Some(visibility) = self.assess_visibility_for_frame(&frame)? {
                // A missing optional reference is not contradictory evidence:
                // calibrated geometry plus current motion can still support a
                // clean non-empty scan. An actual mismatch or inconclusive
                // comparison makes the remembered whole-table layout stale.
                visibility_requires_stale_layout = table_visibility_blocks_full_scan(visibility);
                self.record_visibility_assessment(&frame, visibility, now);
            }
            self.phase_scheduler
                .as_mut()
                .expect("scheduler was checked above")
                .observe_motion(now, evidence)
                .context("phase scheduler rejected motion evidence")?;
            if visibility_requires_stale_layout {
                self.phase_scheduler
                    .as_mut()
                    .expect("scheduler was checked above")
                    .mark_layout_untrusted(now, TrustReason::OccludedOrUncertain)
                    .context("phase scheduler rejected table-appearance evidence")?;
            }
            if evidence != MotionEvidence::Quiet {
                self.cue_geometry = None;
                self.cue_observed_at = None;
                self.cue_observation_frame_id = None;
            }
            if evidence != MotionEvidence::Quiet {
                self.recovery_started_at.get_or_insert(now);
            }

            if self.motion_diagnostics_enabled {
                let phase_after_check = self
                    .phase_scheduler
                    .as_ref()
                    .expect("scheduler was checked above")
                    .phase();
                let monitor_config = self
                    .motion_monitor
                    .as_ref()
                    .expect("motion monitor was checked above")
                    .config();
                let diagnostic = motion_diagnostic_record(MotionDiagnosticRecordInput {
                    session_id: &self.session_id,
                    frame_id: &frame.source().frame_id,
                    observed_at: now,
                    source_width_px: frame.source().width_px,
                    source_height_px: frame.source().height_px,
                    phase_before,
                    phase_after: phase_after_check,
                    assessment,
                    config: monitor_config,
                });
                eprintln!("motion_check={diagnostic}");
            }
            self.finish_scheduled_job(ScheduledJobKind::MotionAssessment, None);
        }

        let phase_after_motion = self
            .phase_scheduler
            .as_ref()
            .expect("scheduler was checked above")
            .phase();
        // Accept cue work only after the current frame's motion evidence has
        // invalidated any older worker result. The estimator owns a second
        // lease over the same budgeted pixels, so this adds no RGB copy and
        // does not keep the coordinator blocked on cue fitting.
        let cue_completion_accepted = self.poll_cue_completion()?;
        let work_now = self.session_elapsed();
        let refreshed_evidence = self
            .phase_scheduler
            .as_ref()
            .expect("scheduler was checked above")
            .evidence(work_now)
            .context("phase scheduler evidence is unavailable after cue completion")?;
        let cue_check_ran = decision.run_cue
            && refreshed_evidence.phase != Phase::Shot
            && self.cue_worker.is_some();
        if cue_check_ran {
            let observed_at = frame
                .host_read_completed_at()
                .checked_duration_since(self.session_clock_origin)
                .unwrap_or_default();
            let observation_age = now.saturating_sub(observed_at);
            let (cue_frame, retained_frame) = frame.into_shared_pair();
            frame = retained_frame;
            let tag = CueWorkTag {
                source_session_id: self.session_id.clone(),
                frame_id: frame.source().frame_id.clone(),
                configuration_revision: self.settings.configuration_revision,
                calibration_revision: self.settings.expected.calibration_revision(),
                motion_generation: refreshed_evidence.motion_generation,
            };
            let worker = self.cue_worker.as_mut().expect("worker checked above");
            match worker.try_submit(tag, cue_frame, &self.cue_proposals, observation_age) {
                Ok(()) => {
                    self.phase_stats.cue_dispatches =
                        self.phase_stats.cue_dispatches.saturating_add(1);
                    self.start_scheduled_job(ScheduledJobKind::CueGeometry);
                }
                Err(error) => {
                    if matches!(
                        error.reason(),
                        crate::cue_worker::CueGeometrySubmitRefusal::Busy
                    ) {
                        self.phase_stats.cue_overload_skips =
                            self.phase_stats.cue_overload_skips.saturating_add(1);
                    }
                    drop(error.into_frame());
                }
            }
        }
        if motion_check_ran
            || cue_check_ran
            || cue_completion_accepted
            || phase_after_motion != phase_before
        {
            if !cue_completion_accepted {
                self.decision_frame_id = Some(frame.source().frame_id.clone());
                self.decision_frame_dimensions =
                    Some((frame.source().width_px, frame.source().height_px));
            }
            self.publish_perception_status(self.session_elapsed())?;
        }

        // A completion may have been waiting in the single-flight worker while
        // this frame's motion check ran. Consume it before submitting today's
        // due frame so a full result slot does not waste the current cadence.
        // This order is intentional: motion above first invalidates any result
        // captured before newly observed movement.
        let completion = self.poll_detector_completion()?;

        if decision.run_ball {
            // Completion processing can change scheduler evidence (for
            // example, by establishing a trusted layout). Re-read it before
            // dispatch and tag this due scan with the current generation. A
            // generation change limits replacement eligibility at completion;
            // it must not cancel measured ball work for the current frame.
            let dispatch_evidence = self
                .phase_scheduler
                .as_ref()
                .context("phase-aware runtime is missing its scheduler")?
                .evidence(self.session_elapsed())
                .context("phase scheduler evidence is unavailable before dispatch")?;
            let tag = DetectorWorkTag {
                source_session_id: self.session_id.clone(),
                frame_id: frame.source().frame_id.clone(),
                configuration_revision: self.settings.configuration_revision,
                calibration_revision: self.settings.expected.calibration_revision(),
                motion_generation: dispatch_evidence.motion_generation,
            };
            let worker = self
                .detector_worker
                .as_mut()
                .context("phase-aware runtime is missing its detector worker")?;
            match worker.try_submit(tag, frame) {
                Ok(()) => {
                    self.phase_stats.detector_calls =
                        self.phase_stats.detector_calls.saturating_add(1);
                    self.start_scheduled_job(ScheduledJobKind::BallDetection);
                    return Ok((CaptureSkipReason::AnalysisNotScheduled, completion));
                }
                Err(error) => {
                    drop(error.into_frame());
                    return Ok((CaptureSkipReason::DetectorOverloaded, completion));
                }
            }
        }
        // Cue refinement above uses this frame independently and never
        // consumes a neural detector call. The lease is dropped here when no
        // full-table inference was due.
        Ok((CaptureSkipReason::AnalysisNotScheduled, completion))
    }

    /// Consume a completed asynchronous cue estimate only when its source and
    /// motion generation are still current. The scheduler receives the frame's
    /// observation time, not worker completion time, so offloading never makes
    /// old cue pixels look fresh.
    fn poll_cue_completion(&mut self) -> Result<bool> {
        let completion = self
            .cue_worker
            .as_mut()
            .and_then(CueGeometryWorker::poll_result);
        let Some(completion) = completion else {
            return Ok(false);
        };
        self.finish_scheduled_job(
            ScheduledJobKind::CueGeometry,
            completion.result.is_err().then_some("cue_geometry_failed"),
        );
        let now = self.session_elapsed();
        let evidence = self
            .phase_scheduler
            .as_ref()
            .context("phase-aware runtime is missing its scheduler")?
            .evidence(now)
            .context("phase scheduler evidence is unavailable for cue result")?;
        if completion.tag.source_session_id != self.session_id
            || completion.tag.configuration_revision != self.settings.configuration_revision
            || completion.tag.calibration_revision != self.settings.expected.calibration_revision()
            || completion.tag.motion_generation != evidence.motion_generation
            || evidence.phase == Phase::Shot
        {
            self.phase_stats.obsolete_results = self.phase_stats.obsolete_results.saturating_add(1);
            return Ok(false);
        }

        let observation_age = cue_observation_age_at_poll(
            completion.timing.observation_age_at_completion,
            completion.timing.processing_completed_at,
            Instant::now(),
        );
        if observation_age
            > self
                .phase_scheduler
                .as_ref()
                .expect("scheduler was checked above")
                .cue_evidence_expiry()
        {
            self.cue_geometry = None;
            self.cue_observed_at = None;
            self.cue_observation_frame_id = None;
            return Ok(false);
        }
        let observed_at = now.saturating_sub(observation_age);
        let frame_id = completion.tag.frame_id.clone();
        let dimensions = (
            completion.frame.source().width_px,
            completion.frame.source().height_px,
        );
        let cue = match completion.result {
            Ok(mut cue) => {
                self.phase_stats.cue_observations =
                    self.phase_stats.cue_observations.saturating_add(1);
                if let (Some(geometry), Some(calibration)) =
                    (cue.as_mut(), self.settings.expected.calibration())
                {
                    let context = CalibrationContext {
                        calibration_revision: self.settings.expected.calibration_revision(),
                        camera_profile_id: self.settings.expected.camera_profile_id(),
                        rectification_revision: self.settings.expected.rectification_revision(),
                        image_space: self.settings.expected.image_space(),
                        source_width_px: dimensions.0,
                        source_height_px: dimensions.1,
                    };
                    orient_cue_geometry_from_playing_surface(geometry, calibration, context);
                }
                cue
            }
            Err(_) => {
                self.phase_stats.cue_failures = self.phase_stats.cue_failures.saturating_add(1);
                None
            }
        };
        let scheduler = self
            .phase_scheduler
            .as_mut()
            .context("phase-aware runtime is missing its scheduler")?;
        let applied = scheduler
            .observe_cue_at(
                now,
                observed_at,
                if cue.is_some() {
                    CueEvidence::Visible
                } else {
                    CueEvidence::NotVisible
                },
            )
            .context("phase scheduler rejected asynchronous cue evidence")?;
        if !applied {
            self.phase_stats.obsolete_results = self.phase_stats.obsolete_results.saturating_add(1);
            return Ok(false);
        }
        self.cue_geometry = cue;
        self.cue_observed_at = Some(observed_at);
        self.cue_observation_frame_id = Some(frame_id.clone());
        self.decision_frame_id = Some(frame_id);
        self.decision_frame_dimensions = Some(dimensions);
        Ok(true)
    }

    fn tick_expiries(&mut self) -> Result<()> {
        let now = self.session_elapsed();
        let evidence_changed = if let Some(scheduler) = self.phase_scheduler.as_mut() {
            let previous = scheduler
                .evidence(now)
                .context("phase scheduler evidence is unavailable before timeout")?;
            // A silent or missing source frame cannot establish quiet, but it
            // must still advance evidence expiry and phase transitions. Do not
            // call `tick` here: a timeout has no frame to process, so consuming
            // due cadence slots would starve work on the next available frame.
            let current = scheduler
                .advance_time(now)
                .context("phase scheduler rejected timeout clock")?;
            previous != current
        } else {
            false
        };
        let cue_expired = self.cue_observed_at.is_some_and(|observed_at| {
            self.phase_scheduler.as_ref().is_some_and(|scheduler| {
                now.saturating_sub(observed_at) > scheduler.cue_evidence_expiry()
            })
        });
        if cue_expired {
            self.cue_geometry = None;
            self.cue_observed_at = None;
            self.cue_observation_frame_id = None;
        }
        if evidence_changed || cue_expired {
            self.publish_perception_status(now)?;
        }
        self.accumulate_trusted_time(now);
        Ok(())
    }

    fn poll_detector_completion(&mut self) -> Result<Option<CapturePipelineOutcome>> {
        let completion = self
            .detector_worker
            .as_mut()
            .and_then(DetectorWorker::poll_result);
        let Some(completion) = completion else {
            return Ok(None);
        };
        self.finish_scheduled_job(
            ScheduledJobKind::BallDetection,
            completion
                .result
                .is_err()
                .then_some("ball_detection_failed"),
        );
        let now = self.session_elapsed();
        let evidence = self
            .phase_scheduler
            .as_ref()
            .context("phase-aware runtime is missing its scheduler")?
            .evidence(now)
            .context("phase scheduler evidence is unavailable")?;
        if completion.tag.source_session_id != self.session_id
            || completion.tag.configuration_revision != self.settings.configuration_revision
            || completion.tag.calibration_revision != self.settings.expected.calibration_revision()
        {
            self.phase_stats.obsolete_results = self.phase_stats.obsolete_results.saturating_add(1);
            return Ok(Some(CapturePipelineOutcome::Skipped(
                CaptureSkipReason::ObsoleteInference,
            )));
        }

        let capture_at = completion
            .frame
            .host_read_completed_at()
            .checked_duration_since(self.session_clock_origin)
            .unwrap_or_default();
        let clean_layout_capture_is_current = self
            .phase_scheduler
            .as_ref()
            .expect("scheduler was checked above")
            .clean_layout_capture_is_current(capture_at, now);
        let within_observation_freshness =
            now.saturating_sub(capture_at) <= self.settings.expected.maximum_frame_age();

        let completion_frame_id = completion.tag.frame_id.clone();
        let completion_frame_dimensions = (
            completion.frame.source().width_px,
            completion.frame.source().height_px,
        );
        let inference = match completion.result {
            Ok(inference) => inference,
            Err(_) => {
                // A failed inference still consumed a specific source frame.
                // Check that frame's appearance before returning so an
                // occluded/uncertain image cannot leave older guidance trusted
                // merely because the detector failed to produce measurements.
                let frame_visibility = self.assess_visibility_for_frame(&completion.frame)?;
                let visibility_telemetry_updated = frame_visibility.is_some_and(|visibility| {
                    self.record_visibility_assessment(&completion.frame, visibility, now)
                });
                let frame_coverage_is_unconfirmed =
                    frame_visibility.is_some_and(table_visibility_blocks_full_scan);
                let newer_unconfirmed_visibility = self
                    .latest_unconfirmed_visibility_capture_at
                    .is_some_and(|observed_at| observed_at > capture_at);
                let visibility_requires_stale_layout =
                    frame_coverage_is_unconfirmed || newer_unconfirmed_visibility;
                if visibility_requires_stale_layout {
                    self.phase_scheduler
                        .as_mut()
                        .expect("scheduler was checked above")
                        .mark_layout_untrusted(now, TrustReason::OccludedOrUncertain)
                        .context("phase scheduler rejected failed-frame appearance evidence")?;
                }
                if visibility_requires_stale_layout || visibility_telemetry_updated {
                    self.decision_frame_id = Some(completion_frame_id.clone());
                    self.decision_frame_dimensions = Some(completion_frame_dimensions);
                    self.publish_perception_status(now)?;
                }
                self.phase_stats.detector_failures =
                    self.phase_stats.detector_failures.saturating_add(1);
                self.phase_scheduler
                    .as_mut()
                    .expect("scheduler was checked above")
                    .observe_layout(now, LayoutObservation::Failed)
                    .context("phase scheduler rejected failed detector result")?;
                return Ok(Some(CapturePipelineOutcome::Skipped(
                    CaptureSkipReason::DetectorFailed,
                )));
            }
        };
        // An empty detector result may clear the layout only when the exact
        // frame that produced it positively showed the calibrated table. The
        // latest motion/appearance assessment can belong to a newer frame
        // while this detector call was in flight.
        let inference_visibility = self.assess_visibility_for_frame(&completion.frame)?;
        let visibility_telemetry_updated = inference_visibility.is_some_and(|visibility| {
            self.record_visibility_assessment(&completion.frame, visibility, now)
        });
        let inference_is_not_visible = inference_visibility
            .is_some_and(|visibility| visibility.status == TableVisibilityStatus::NotVisible);
        let inference_confirms_full_table_coverage = inference_visibility
            .is_some_and(|visibility| visibility.status == TableVisibilityStatus::Visible);
        // Every wholesale replacement needs positive evidence that this exact
        // frame covers the calibrated table. Missing monitor/reference data is
        // inconclusive: current detections still publish as partial results,
        // while unmatched last-good centers remain available.
        let inference_coverage_is_unconfirmed = !inference_confirms_full_table_coverage;
        // Keep contradictory capture times separately from the most recently
        // reported assessment. A later visible frame may update diagnostics,
        // but cannot make an older in-flight scan eligible to replace the
        // layout after a newer uncertain or NotVisible frame was captured.
        let newer_unconfirmed_visibility = self
            .latest_unconfirmed_visibility_capture_at
            .is_some_and(|observed_at| observed_at > capture_at);
        let newer_not_visible = self
            .latest_not_visible_capture_at
            .is_some_and(|observed_at| observed_at > capture_at);
        let appearance_conflicts_with_table = inference_is_not_visible || newer_not_visible;
        let visibility_coverage_is_unconfirmed =
            inference_coverage_is_unconfirmed || newer_unconfirmed_visibility;
        let visibility_requires_stale_layout = visibility_coverage_is_unconfirmed;
        if visibility_requires_stale_layout {
            // A contradictory or inconclusive frame can arrive between motion
            // checks or while inference is in flight. Make the remembered map
            // stale before publication, but keep any current measurements.
            self.phase_scheduler
                .as_mut()
                .expect("scheduler was checked above")
                .mark_layout_untrusted(now, TrustReason::OccludedOrUncertain)
                .context("phase scheduler rejected completed appearance evidence")?;
        }
        if visibility_requires_stale_layout || visibility_telemetry_updated {
            self.decision_frame_id = Some(completion_frame_id.clone());
            self.decision_frame_dimensions = Some(completion_frame_dimensions);
            // Publish the invalidated (or explicitly uncertain) evidence
            // before any partial scene is serialized. The pipeline carries
            // its latest perception state into the next scene envelope, so
            // that first wire scene cannot briefly revive stale guidance.
            self.publish_perception_status(now)?;
        }
        // Motion generations and the quiet dwell control whether a scan can
        // replace the trusted whole-table layout. Newer appearance evidence
        // also forbids replacement, but never discards fresh measured centers
        // from the detector frame.
        let can_replace_layout = clean_layout_capture_is_current
            && completion.tag.motion_generation == evidence.motion_generation
            && inference_confirms_full_table_coverage
            && !visibility_coverage_is_unconfirmed;
        let contains_ball_detection = self.pipeline.contains_ball_detection(&inference);
        if stale_limited_observation_is_obsolete(
            can_replace_layout,
            appearance_conflicts_with_table,
            within_observation_freshness,
        ) {
            self.phase_stats.obsolete_results = self.phase_stats.obsolete_results.saturating_add(1);
            return Ok(Some(CapturePipelineOutcome::Skipped(
                CaptureSkipReason::ObsoleteInference,
            )));
        }
        if !contains_ball_detection
            && (!inference_confirms_full_table_coverage || newer_unconfirmed_visibility)
        {
            // A clean empty result is authoritative only when the table was
            // positively visible. An explicitly unknown appearance is not a
            // clear-table observation, even if motion evidence otherwise
            // permits a full scan. Phase-aware empty replacement always
            // requires positive coverage evidence, including when no optional
            // appearance monitor was configured for this session.
            self.phase_stats.visibility_rejected_empty_scans = self
                .phase_stats
                .visibility_rejected_empty_scans
                .saturating_add(1);
            return Ok(Some(CapturePipelineOutcome::Skipped(
                CaptureSkipReason::UnverifiedEmptyLayout,
            )));
        }
        self.cue_proposals =
            cue_proposals_from_inference(&inference, self.cue_config.max_proposals);
        let outcome = if !can_replace_layout {
            self.pipeline.process_partial_inference_result(
                &completion.frame,
                inference,
                completion.inference_started,
                completion.inference_completed,
            )
        } else {
            self.pipeline.process_inference_result(
                &completion.frame,
                inference,
                completion.inference_started,
                completion.inference_completed,
            )
        };
        match outcome {
            Ok(outcome @ CapturePipelineOutcome::Published(_)) => {
                let CapturePipelineOutcome::Published(result) = &outcome else {
                    unreachable!();
                };
                let positions = result
                    .resolved_ball_measurements
                    .measurements
                    .iter()
                    .map(|measurement| ImagePoint {
                        x: measurement.center_image_px[0],
                        y: measurement.center_image_px[1],
                    })
                    .collect();
                let was_trusted = evidence.map_trusted;
                if can_replace_layout && !appearance_conflicts_with_table {
                    self.phase_scheduler
                        .as_mut()
                        .expect("scheduler was checked above")
                        .observe_layout_at(now, capture_at, LayoutObservation::Success(positions))
                        .context("phase scheduler rejected successful layout")?;
                }
                if can_replace_layout
                    && self
                        .phase_scheduler
                        .as_ref()
                        .expect("scheduler was checked above")
                        .map_trusted()
                {
                    // Bind trust restoration to this exact clean scan. The
                    // scene write may expire during reconnect while its later
                    // phase-only status still reaches the gateway.
                    self.layout_basis_frame_id = Some(completion_frame_id.clone());
                }
                if !was_trusted
                    && self
                        .phase_scheduler
                        .as_ref()
                        .expect("scheduler was checked above")
                        .map_trusted()
                {
                    if let Some(monitor) = self.motion_monitor.as_mut() {
                        monitor
                            .set_current_as_verified_reference()
                            .context("failed to promote verified layout motion reference")?;
                    }
                    if let Some(started) = self.recovery_started_at.take() {
                        self.phase_stats.last_recovery_delay_ms = Some(
                            now.saturating_sub(started)
                                .as_millis()
                                .min(u128::from(u64::MAX)) as u64,
                        );
                    }
                }
                self.decision_frame_id = Some(completion_frame_id);
                self.decision_frame_dimensions = Some(completion_frame_dimensions);
                self.publish_perception_status(now)?;
                Ok(Some(outcome))
            }

            Ok(outcome) => {
                self.phase_scheduler
                    .as_mut()
                    .expect("scheduler was checked above")
                    .observe_layout(now, LayoutObservation::Failed)
                    .context("phase scheduler rejected skipped layout")?;
                Ok(Some(outcome))
            }
            Err(error) => {
                self.phase_scheduler
                    .as_mut()
                    .expect("scheduler was checked above")
                    .observe_layout(now, LayoutObservation::Failed)
                    .context("phase scheduler rejected invalid layout")?;
                Err(error)
            }
        }
    }

    /// Assess appearance on the supplied capture pixels rather than borrowing
    /// a decision made for another frame. This is used both for current-frame
    /// motion diagnostics and, critically, when deciding whether an empty
    /// detector result can replace the cached layout.
    fn assess_visibility_for_frame(
        &self,
        frame: &OwnedCaptureFrame,
    ) -> Result<Option<TableVisibilityAssessment>> {
        let Some(config) = self.table_visibility_config.as_ref() else {
            return Ok(None);
        };
        let mut visibility = assess_table_visibility(
            self.table_visibility_reference.as_ref(),
            Some(VisibilityRgb8Frame {
                width: frame.source().width_px,
                height: frame.source().height_px,
                pixels: frame.rgb8(),
            }),
            config,
        )
        .context("calibrated-table visibility assessment failed")?;
        if matches!(
            visibility.reason,
            TableVisibilityReason::ExtremeBrightnessShift
                | TableVisibilityReason::ExtremeColorShift
        ) {
            // The frame is available, but an illumination mismatch makes the
            // stored appearance reference inconclusive rather than proving
            // that the table is absent.
            visibility.status = TableVisibilityStatus::Unknown;
        }
        Ok(Some(visibility))
    }

    fn record_visibility_assessment(
        &mut self,
        frame: &OwnedCaptureFrame,
        assessment: TableVisibilityAssessment,
        observed_at: Duration,
    ) -> bool {
        let captured_at = frame
            .host_read_completed_at()
            .checked_duration_since(self.session_clock_origin)
            .unwrap_or_default();

        // Visibility telemetry describes the newest source frame, even when
        // an older detector completion is assessed after newer frames.
        let telemetry_updated = self
            .latest_visibility_capture_at
            .is_none_or(|latest| captured_at > latest);
        if telemetry_updated {
            self.latest_visibility = Some((frame.source().frame_id.clone(), assessment));
            self.latest_visibility_observed_at = Some(observed_at);
            self.latest_visibility_capture_at = Some(captured_at);
        }

        if table_visibility_blocks_full_scan(assessment) {
            self.latest_unconfirmed_visibility_capture_at = Some(
                self.latest_unconfirmed_visibility_capture_at
                    .map_or(captured_at, |latest| latest.max(captured_at)),
            );
        }
        if assessment.status == TableVisibilityStatus::NotVisible {
            self.latest_not_visible_capture_at = Some(
                self.latest_not_visible_capture_at
                    .map_or(captured_at, |latest| latest.max(captured_at)),
            );
        }
        telemetry_updated
    }

    fn accumulate_trusted_time(&mut self, now: Duration) {
        if let (Some(previous), Some(scheduler)) =
            (self.last_scheduler_tick, self.phase_scheduler.as_ref())
        {
            if scheduler.map_trusted() {
                self.phase_stats.trusted_layout_duration_ms =
                    self.phase_stats.trusted_layout_duration_ms.saturating_add(
                        now.saturating_sub(previous)
                            .as_millis()
                            .min(u128::from(u64::MAX)) as u64,
                    );
            }
        }
        self.last_scheduler_tick = Some(now);
    }

    fn invalidate_phase_context(&mut self) {
        let now = self.session_elapsed();
        self.cue_geometry = None;
        self.cue_observed_at = None;
        self.cue_observation_frame_id = None;
        if let Some(scheduler) = self.phase_scheduler.as_mut() {
            let _ = scheduler.invalidate_context(now);
        }
        if let Some(monitor) = self.motion_monitor.as_mut() {
            monitor.reset();
        }
    }

    fn session_elapsed(&self) -> Duration {
        self.session_clock_origin.elapsed()
    }

    fn publish_perception_status(&mut self, now: Duration) -> Result<()> {
        let (Some(config), Some(scheduler)) = (
            self.phase_schedule_config.as_ref(),
            self.phase_scheduler.as_ref(),
        ) else {
            return Ok(());
        };
        let evidence = scheduler
            .evidence(now)
            .context("phase scheduler evidence is unavailable for publication")?;
        let age_ms = |observed_at: Option<Duration>| {
            observed_at
                .map(|at| now.saturating_sub(at).as_millis().min(u128::from(u64::MAX)) as u64)
        };
        let cue = self.cue_geometry().map(|geometry| CueGeometryStateV1 {
            shaft_start_px: geometry.shaft_start_px,
            shaft_end_px: geometry.shaft_end_px,
            axis_unit: geometry.axis_unit,
            confidence: geometry.confidence,
            observation_age_ms: geometry.observation_age_ms,
            signed_direction: geometry.signed_direction.map(|direction| match direction {
                SignedCueDirection::TowardStart => "toward_start".to_owned(),
                SignedCueDirection::TowardEnd => "toward_end".to_owned(),
            }),
            source_frame_id: self.cue_observation_frame_id.clone(),
        });
        let visibility = self
            .latest_visibility
            .as_ref()
            .map(|(frame_id, assessment)| TableVisibilityStateV1 {
                status: match assessment.status {
                    TableVisibilityStatus::Visible => TableVisibilityStatusV1::Visible,
                    TableVisibilityStatus::NotVisible => TableVisibilityStatusV1::NotVisible,
                    TableVisibilityStatus::Unknown => TableVisibilityStatusV1::Unknown,
                },
                reason: table_visibility_reason_name(assessment.reason).to_owned(),
                observation_frame_id: frame_id.clone(),
                observation_age_ms: age_ms(self.latest_visibility_observed_at).unwrap_or(u64::MAX),
            });
        let perception = PerceptionStateV1 {
            event_sequence: 0,
            image_width_px: self
                .decision_frame_dimensions
                .map(|dimensions| dimensions.0),
            image_height_px: self
                .decision_frame_dimensions
                .map(|dimensions| dimensions.1),
            phase: phase_name(evidence.phase).to_owned(),
            layout_revision: evidence.layout_revision,
            layout_trusted: evidence.map_trusted,
            layout_basis_frame_id: self.layout_basis_frame_id.clone(),
            empty_layout_verified: evidence.empty_layout_verified,
            guidance_valid: evidence.guidance_valid && cue.is_some(),
            trust_reason: trust_reason_name(evidence.trust_reason).to_owned(),
            last_ball_observation_age_ms: age_ms(evidence.layout_at),
            latest_motion_check_age_ms: age_ms(evidence.motion_at),
            cue_evidence_age_ms: age_ms(evidence.cue_at),
            motion_evidence_max_age_ms: millis(config.motion_evidence_expiry),
            cue_evidence_max_age_ms: millis(config.cue_evidence_expiry),
            layout_verification_max_age_ms: millis(config.layout_evidence_expiry),
            cue,
            decision_frame_id: self.decision_frame_id.clone(),
            decision_calibration_revision: Some(self.settings.expected.calibration_revision()),
            decision_rectification_revision: Some(self.settings.expected.rectification_revision()),
            table_visibility: visibility,
            table_projection: self
                .settings
                .expected
                .calibration()
                .map(table_projection_from_calibration),
        };
        self.pipeline
            .publish_perception_state(&self.session_id, perception, Instant::now())?;
        Ok(())
    }
}

fn cue_observation_age_at_poll(
    observation_age_at_completion: Duration,
    processing_completed_at: Instant,
    polled_at: Instant,
) -> Duration {
    observation_age_at_completion
        .saturating_add(polled_at.saturating_duration_since(processing_completed_at))
}

pub(crate) fn table_projection_from_calibration(
    table: &ValidatedTableCalibrationV1,
) -> TableProjectionV1 {
    let (table_width_mm, table_height_mm) = table.playing_surface_dimensions_mm();
    let (source_width_px, source_height_px) = table.source_dimensions_px();
    TableProjectionV1 {
        direction: "source_image_px_to_table_mm".to_owned(),
        image_space: format!("{:?}", table.image_space()).to_lowercase(),
        source_width_px,
        source_height_px,
        camera_profile_id: table.camera_profile_id().to_owned(),
        calibration_revision: table.calibration_revision(),
        rectification_revision: table.rectification_revision(),
        table_width_mm,
        table_height_mm,
        unit: "millimeters".to_owned(),
        origin: "playing_surface_center".to_owned(),
        x_axis: "corner_0_to_1".to_owned(),
        y_axis: "corner_0_to_3".to_owned(),
        pixel_to_table_mm: table.pixel_to_table_mm(),
    }
}

fn cue_proposals_from_inference(
    inference: &crate::inference::FrameInferenceOutput,
    maximum: usize,
) -> Vec<CueProposal> {
    let mut proposals = Vec::with_capacity(maximum.min(inference.detections.len()));
    for detection in &inference.detections {
        if proposals.len() >= maximum || !detection.class_name.trim().eq_ignore_ascii_case("cue") {
            continue;
        }
        let [left, top, right, bottom] = detection.bbox_xyxy_px;
        if ![left, top, right, bottom]
            .iter()
            .all(|value| value.is_finite())
        {
            continue;
        }
        let width = inference.source_width_px as f32;
        let height = inference.source_height_px as f32;
        let left = left.floor().clamp(0.0, width) as u32;
        let top = top.floor().clamp(0.0, height) as u32;
        let right = right.ceil().clamp(0.0, width) as u32;
        let bottom = bottom.ceil().clamp(0.0, height) as u32;
        if right > left && bottom > top {
            proposals.push(CueProposal {
                left,
                top,
                right,
                bottom,
            });
        }
    }
    proposals
}

fn millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn scheduler_phase_is_gated(config: Option<&SchedulerConfig>, phase: Phase) -> bool {
    let Some(config) = config else {
        return false;
    };
    let cadence = match phase {
        Phase::Recovering => config.recovering,
        Phase::PreShot => config.pre_shot,
        Phase::Aiming => config.aiming,
        Phase::Shot => config.shot,
    };
    cadence.motion_hz == 0.0 && cadence.ball_hz == 0.0 && cadence.cue_hz == 0.0
}

fn table_visibility_reason_name(reason: TableVisibilityReason) -> &'static str {
    match reason {
        TableVisibilityReason::ReferenceMatch => "reference_match",
        TableVisibilityReason::MissingReference => "missing_reference",
        TableVisibilityReason::MissingFrame => "missing_frame",
        TableVisibilityReason::InvalidReference => "invalid_reference",
        TableVisibilityReason::SamplingLayoutChanged => "sampling_layout_changed",
        TableVisibilityReason::ResolutionChanged => "resolution_changed",
        TableVisibilityReason::ExtremeBrightnessShift => "extreme_brightness_shift",
        TableVisibilityReason::ExtremeColorShift => "extreme_color_shift",
        TableVisibilityReason::LostRailContrast => "lost_rail_contrast",
        TableVisibilityReason::ClothMismatch => "cloth_mismatch",
        TableVisibilityReason::RailMismatch => "rail_mismatch",
    }
}

/// Slow clean scans can still verify a quiet layout. Once motion or table
/// appearance limits a result to partial use, its centers must also fit the
/// configured capture-age budget.
fn stale_limited_observation_is_obsolete(
    can_replace_layout: bool,
    appearance_conflicts_with_table: bool,
    within_observation_freshness: bool,
) -> bool {
    !within_observation_freshness && (!can_replace_layout || appearance_conflicts_with_table)
}

/// Without a positive appearance assessment, the runtime cannot prove that a
/// scan covered the whole table. Such scans remain useful as partial measured
/// observations but cannot replace unmatched last-good centers.
fn table_visibility_blocks_full_scan(assessment: TableVisibilityAssessment) -> bool {
    assessment.status != TableVisibilityStatus::Visible
}

fn phase_name(phase: Phase) -> &'static str {
    match phase {
        Phase::Recovering => "recovering",
        Phase::PreShot => "pre_shot",
        Phase::Aiming => "aiming",
        Phase::Shot => "shot_underway",
    }
}

fn trust_reason_name(reason: TrustReason) -> &'static str {
    match reason {
        TrustReason::Trusted => "trusted",
        TrustReason::AwaitingSettlement => "awaiting_settlement",
        TrustReason::MotionDetected => "motion_detected",
        TrustReason::OccludedOrUncertain => "occluded_or_uncertain",
        TrustReason::MotionEvidenceExpired => "motion_evidence_expired",
        TrustReason::LayoutVerificationExpired => "layout_verification_expired",
        TrustReason::ContextChanged => "context_changed",
    }
}

/// Build one structured record from the exact assessment used by the phase
/// scheduler. Keeping the classifier metrics and effective thresholds beside
/// the frame identity makes live false positives diagnosable without replaying
/// an approximate preview stream.
struct MotionDiagnosticRecordInput<'a> {
    session_id: &'a str,
    frame_id: &'a str,
    observed_at: Duration,
    source_width_px: u32,
    source_height_px: u32,
    phase_before: Phase,
    phase_after: Phase,
    assessment: MotionAssessment,
    config: &'a MotionMonitorConfig,
}

fn motion_diagnostic_record(input: MotionDiagnosticRecordInput<'_>) -> serde_json::Value {
    let MotionDiagnosticRecordInput {
        session_id,
        frame_id,
        observed_at,
        source_width_px,
        source_height_px,
        phase_before,
        phase_after,
        assessment,
        config,
    } = input;
    let sample_count =
        u64::from(assessment.sample_width).saturating_mul(u64::from(assessment.sample_height));
    let changed_samples = |fraction: Option<f64>| {
        fraction.map(|fraction| (fraction * sample_count as f64).round() as u64)
    };
    serde_json::json!({
        "event": "motion_check",
        "session_id": session_id,
        "frame_id": frame_id,
        "session_elapsed_ms": millis(observed_at),
        "source_width_px": source_width_px,
        "source_height_px": source_height_px,
        "phase_before": phase_name(phase_before),
        "phase_after": phase_name(phase_after),
        "classification": motion_classification_name(assessment.classification),
        "reason": motion_reason_name(assessment.reason),
        "sample_width": assessment.sample_width,
        "sample_height": assessment.sample_height,
        "sample_count": sample_count,
        "consecutive_changed_fraction": assessment.consecutive_changed_fraction,
        "consecutive_changed_samples": changed_samples(Some(assessment.consecutive_changed_fraction)),
        "reference_changed_fraction": assessment.reference_changed_fraction,
        "reference_changed_samples": changed_samples(assessment.reference_changed_fraction),
        "reference_change_delta_fraction": assessment.reference_change_delta_fraction,
        "reference_change_delta_samples": changed_samples(assessment.reference_change_delta_fraction),
        "consecutive_brightness_shift": assessment.consecutive_brightness_shift,
        "reference_brightness_shift": assessment.reference_brightness_shift,
        "luminance_difference_threshold": config.luminance_difference_threshold,
        "movement_fraction_threshold": config.movement_fraction_threshold,
        "broad_change_fraction_threshold": config.broad_change_fraction,
        "broad_change_extent_fraction_threshold": config.broad_change_extent_fraction,
        "broad_change_min_fraction_threshold": config.broad_change_min_fraction,
        "maximum_global_brightness_shift": config.maximum_global_brightness_shift,
    })
}

fn motion_classification_name(classification: MotionClassification) -> &'static str {
    match classification {
        MotionClassification::Quiet => "quiet",
        MotionClassification::Moving => "moving",
        MotionClassification::OccludedOrUncertain => "occluded_or_uncertain",
    }
}

fn motion_reason_name(reason: MotionReason) -> &'static str {
    match reason {
        MotionReason::Initializing => "initializing",
        MotionReason::Stable => "stable",
        MotionReason::ConsecutiveChange => "consecutive_change",
        MotionReason::ReferenceDrift => "reference_drift",
        MotionReason::BroadChange => "broad_change",
        MotionReason::ExtremeBrightnessShift => "extreme_brightness_shift",
        MotionReason::ResolutionChanged => "resolution_changed",
    }
}

/// Counts actual work and skips in a phase-aware live session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PhaseRuntimeStats {
    pub captured_frames: u64,
    pub motion_checks: u64,
    pub detector_calls: u64,
    pub detector_failures: u64,
    pub cue_observations: u64,
    pub cue_dispatches: u64,
    pub cue_failures: u64,
    pub cue_overload_skips: u64,
    pub intentional_skips: u64,
    pub overload_skips: u64,
    pub obsolete_results: u64,
    pub visibility_rejected_empty_scans: u64,
    pub trusted_layout_duration_ms: u64,
    pub last_recovery_delay_ms: Option<u64>,
}

/// Actual work counters and achieved rates over the live session window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PhaseRuntimeDiagnostics {
    pub elapsed_ms: u64,
    pub captured_fps: Option<f64>,
    pub motion_checks_per_second: Option<f64>,
    pub detector_calls_per_second: Option<f64>,
    pub cue_observations_per_second: Option<f64>,
    pub work: PhaseRuntimeStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cue_age_includes_time_waiting_after_worker_completion() {
        let completed = Instant::now();
        let polled_later = completed + Duration::from_millis(100);
        assert_eq!(
            cue_observation_age_at_poll(Duration::from_millis(200), completed, polled_later,),
            Duration::from_millis(300),
        );
        assert!(
            cue_observation_age_at_poll(Duration::from_millis(200), completed, polled_later,)
                > Duration::from_millis(250)
        );
    }

    #[test]
    fn stale_partial_results_are_rejected_without_rejecting_slow_clean_scans() {
        assert!(stale_limited_observation_is_obsolete(true, true, false));
        assert!(stale_limited_observation_is_obsolete(false, false, false));
        assert!(!stale_limited_observation_is_obsolete(true, false, false));
        assert!(!stale_limited_observation_is_obsolete(false, true, true));
    }

    #[test]
    fn public_projection_metadata_is_the_active_revision_bound_homography() {
        let calibration: crate::calibration::TableCalibrationV1 = serde_json::from_str(
            include_str!("../contracts/calibration/v1/table-calibration-perspective.json"),
        )
        .unwrap();
        let calibration = calibration.into_validated().unwrap();
        let projection = table_projection_from_calibration(&calibration);
        assert_eq!(projection.direction, "source_image_px_to_table_mm");
        assert_eq!(projection.calibration_revision, 3);
        assert_eq!(projection.rectification_revision, 2);
        assert_eq!(
            (projection.source_width_px, projection.source_height_px),
            (1000, 500)
        );
        assert_eq!(
            (projection.table_width_mm, projection.table_height_mm),
            (2000.0, 1000.0)
        );
        assert_eq!(
            projection.pixel_to_table_mm,
            calibration.pixel_to_table_mm()
        );
        assert_eq!(projection.origin, "playing_surface_center");
    }

    #[test]
    fn motion_diagnostic_record_includes_the_trigger_and_effective_thresholds() {
        let config = MotionMonitorConfig::default();
        let assessment = MotionAssessment {
            classification: MotionClassification::Moving,
            reason: MotionReason::ConsecutiveChange,
            consecutive_changed_fraction: 8.0 / (160.0 * 90.0),
            reference_changed_fraction: Some(0.01),
            reference_change_delta_fraction: Some(0.002),
            consecutive_brightness_shift: 2,
            reference_brightness_shift: Some(-1),
            sample_width: 160,
            sample_height: 90,
        };

        let record = motion_diagnostic_record(MotionDiagnosticRecordInput {
            session_id: "session-1",
            frame_id: "frame-42",
            observed_at: Duration::from_millis(1234),
            source_width_px: 1920,
            source_height_px: 1080,
            phase_before: Phase::PreShot,
            phase_after: Phase::Shot,
            assessment,
            config: &config,
        });

        assert_eq!(record["event"], "motion_check");
        assert_eq!(record["session_id"], "session-1");
        assert_eq!(record["frame_id"], "frame-42");
        assert_eq!(record["phase_before"], "pre_shot");
        assert_eq!(record["phase_after"], "shot_underway");
        assert_eq!(record["classification"], "moving");
        assert_eq!(record["reason"], "consecutive_change");
        assert_eq!(record["sample_count"], 14_400);
        assert_eq!(record["consecutive_changed_samples"], 8);
        assert_eq!(
            record["movement_fraction_threshold"],
            config.movement_fraction_threshold
        );
        assert_eq!(
            record["luminance_difference_threshold"],
            config.luminance_difference_threshold
        );
    }
}
