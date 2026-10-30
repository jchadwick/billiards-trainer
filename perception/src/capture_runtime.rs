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
use crate::cue_worker::{CueGeometryWorker, CueGeometryWorkerShutdown, CueWorkTag};
use crate::detector_worker::{DetectorWorkTag, DetectorWorker, DetectorWorkerShutdown};
use crate::frame_preview::LatestCapturedFrameSlot;
use crate::live_audit::LiveAuditRecorder;
use crate::motion_monitor::{
    MotionAssessment, MotionClassification, MotionMonitor, MotionMonitorConfig, MotionReason,
};
use crate::phase_scheduler::{
    CueEvidence, ImagePoint, LayoutObservation, MotionEvidence, Phase, PhaseScheduler,
    SchedulerConfig, SchedulerEvidence, TrustReason, WorkAvailability,
};
use crate::public_state_lifecycle::{
    CueGeometryStateV1, PerceptionStateV1, TableVisibilityStateV1, TableVisibilityStatusV1,
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
    recovery_started_at: Option<Duration>,
    table_visibility_config: Option<TableVisibilityConfig>,
    table_visibility_reference: Option<TableVisibilityReference>,
    visible_reacquisition_samples: usize,
    /// True only after the most recent accepted ball layout supplied a table
    /// position for every visible ball through the same refiner/calibration.
    latest_precision_ready: bool,
    latest_visibility: Option<(String, TableVisibilityAssessment)>,
    latest_visibility_observed_at: Option<Duration>,
    decision_frame_id: Option<String>,
    decision_frame_dimensions: Option<(u32, u32)>,
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
            recovery_started_at: None,
            table_visibility_config,
            table_visibility_reference,
            visible_reacquisition_samples: 0,
            latest_precision_ready: false,
            latest_visibility: None,
            latest_visibility_observed_at: None,
            decision_frame_id: None,
            decision_frame_dimensions: None,
            receiver: None,
        })
    }

    /// Current readiness state. A timeout leaves this at `Pending` and keeps
    /// the source coordinator and its owner gate in this runtime.
    pub fn lifecycle(&self) -> &LiveCaptureLifecycle {
        &self.lifecycle
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
                let (no_dispatch, completion) = self.observe_and_schedule_frame(frame)?;
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
        self.accumulate_trusted_time(now);
        let motion_check_ran = decision.run_motion;

        if decision.run_motion {
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
            let mut evidence = match assessment.classification {
                MotionClassification::Quiet => MotionEvidence::Quiet,
                MotionClassification::Moving => MotionEvidence::Moving,
                MotionClassification::OccludedOrUncertain => MotionEvidence::OccludedOrUncertain,
            };
            if let Some(config) = self.table_visibility_config.as_ref() {
                let visibility = assess_table_visibility(
                    self.table_visibility_reference.as_ref(),
                    Some(VisibilityRgb8Frame {
                        width: frame.source().width_px,
                        height: frame.source().height_px,
                        pixels: frame.rgb8(),
                    }),
                    config,
                )
                .context("calibrated-table visibility assessment failed")?;
                if visibility.status == TableVisibilityStatus::Visible {
                    self.visible_reacquisition_samples =
                        self.visible_reacquisition_samples.saturating_add(1);
                    if self.visible_reacquisition_samples < config.reacquisition_consensus_count {
                        evidence = MotionEvidence::OccludedOrUncertain;
                    }
                } else {
                    self.visible_reacquisition_samples = 0;
                    evidence = MotionEvidence::OccludedOrUncertain;
                }
                self.latest_visibility = Some((frame.source().frame_id.clone(), visibility));
                self.latest_visibility_observed_at = Some(now);
            }
            self.phase_scheduler
                .as_mut()
                .expect("scheduler was checked above")
                .observe_motion(now, evidence)
                .context("phase scheduler rejected motion evidence")?;
            if evidence != MotionEvidence::Quiet {
                self.latest_precision_ready = false;
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
            // dispatch so a stale pre-completion decision cannot cross a new
            // motion generation or submit after the scheduler entered Shot.
            let dispatch_evidence = self
                .phase_scheduler
                .as_ref()
                .context("phase-aware runtime is missing its scheduler")?
                .evidence(self.session_elapsed())
                .context("phase scheduler evidence is unavailable before dispatch")?;
            if dispatch_evidence.motion_generation != decision.motion_generation {
                return Ok((CaptureSkipReason::AnalysisNotScheduled, completion));
            }
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

        let observation_age = completion.timing.observation_age_at_completion;
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
        if evidence_changed {
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
            || completion.tag.motion_generation != evidence.motion_generation
        {
            self.latest_precision_ready = false;
            self.phase_stats.obsolete_results = self.phase_stats.obsolete_results.saturating_add(1);
            return Ok(Some(CapturePipelineOutcome::Skipped(
                CaptureSkipReason::ObsoleteInference,
            )));
        }

        let completion_frame_id = completion.tag.frame_id.clone();
        let completion_frame_dimensions = (
            completion.frame.source().width_px,
            completion.frame.source().height_px,
        );
        let inference = match completion.result {
            Ok(inference) => inference,
            Err(_) => {
                self.latest_precision_ready = false;
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
        self.cue_proposals =
            cue_proposals_from_inference(&inference, self.cue_config.max_proposals);
        let outcome = self.pipeline.process_inference_result(
            &completion.frame,
            inference,
            completion.inference_started,
            completion.inference_completed,
        );
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
                self.latest_precision_ready =
                    result
                        .resolved_ball_measurements
                        .calibration_revision
                        .is_some()
                        && result.resolved_ball_measurements.measurements.iter().all(
                            |measurement| {
                                measurement.refinement_status
                                    == Some(crate::ball_localization::RefinementStatus::Accepted)
                                    && measurement.table_center_mm.is_some()
                            },
                        );
                let was_trusted = evidence.map_trusted;
                self.phase_scheduler
                    .as_mut()
                    .expect("scheduler was checked above")
                    .observe_layout(now, LayoutObservation::Success(positions))
                    .context("phase scheduler rejected successful layout")?;
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
                self.latest_precision_ready = false;
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
        let visibility_is_current = visibility.as_ref().is_some_and(|visibility| {
            visibility.status == TableVisibilityStatusV1::Visible
                && visibility.observation_age_ms <= millis(config.motion_evidence_expiry)
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
            empty_layout_verified: evidence.empty_layout_verified,
            guidance_valid: evidence.guidance_valid
                && visibility_is_current
                && self.latest_precision_ready
                && self.decision_frame_id.is_some()
                && self.cue_observation_frame_id.as_deref() == self.decision_frame_id.as_deref(),
            trust_reason: if evidence.guidance_valid && !self.latest_precision_ready {
                "precise_ball_positions_unavailable".to_owned()
            } else {
                trust_reason_name(evidence.trust_reason).to_owned()
            },
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
        };
        self.pipeline
            .publish_perception_state(&self.session_id, perception, Instant::now())?;
        Ok(())
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
