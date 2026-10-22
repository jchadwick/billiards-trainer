//! Rust-owned live capture lifecycle.
//!
//! This module composes the strict Python capture-settings request, exclusive
//! source coordinator, RGB8 scene pipeline and request-bound v3 receiver
//! handshake. A frame receiver is withheld until source preflight and the v3
//! readiness exchange both succeed. Timeout and failure states retain the
//! coordinator so callers can explicitly stop and confirm source release.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};

use crate::calibration::ValidatedTableCalibrationV1;
use crate::capture::{
    CaptureFrameReceiver, CaptureOwnerDisposition, CaptureOwnerStatus, CapturePreflightStatus,
    CapturePreflightWait, CaptureReceiveOutcome, CaptureRuntimeCoordinator, CaptureRuntimeWait,
    CaptureSourceFactory, CaptureWorkerReport,
};
use crate::capture_pipeline::{
    CapturePipelineConfig, CapturePipelineOutcome, CapturePipelineStats, CaptureScenePipeline,
    Rgb8FrameDetector,
};
use crate::capture_settings_wire::{parse_capture_settings_json, ResolvedCaptureSettings};
use crate::frame_preview::LatestCapturedFrameSlot;
use crate::live_audit::LiveAuditRecorder;
use crate::scene_transport::SceneTransportStats;
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
/// Construction accepts the exact UTF-8 request bytes produced by Python.
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
            .finish_non_exhaustive()
    }
}

impl<D: Rgb8FrameDetector> LiveCaptureRuntime<D> {
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
        source_factory: F,
    ) -> Result<Self> {
        let mut settings = parse_capture_settings_json(exact_request_json)
            .context("invalid exact Python capture-settings request")?;
        if let Some((revision, calibration)) = calibration_store_state {
            settings
                .expected
                .apply_calibration_store_state(revision, calibration)
                .context("Rust-owned calibration store state does not match the capture context")?;
        }

        // Establish the pipeline clock origin and its bounded scene writer
        // before the capture worker is permitted to open its source.
        let pipeline =
            CaptureScenePipeline::new_with_audit(detector, tracking, pipeline_config, audit)
                .context("failed to create live capture-to-scene pipeline")?;
        pipeline
            .validate_expected_capture_settings(&settings.expected)
            .map_err(|error| {
                anyhow!("pipeline configuration disagrees with capture request: {error}")
            })?;

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
        let receiver = self
            .receiver
            .as_mut()
            .context("ready runtime does not own its frame receiver")?;
        let outcome = self.pipeline.receive_one(receiver, timeout);
        match &outcome {
            CaptureReceiveOutcome::Frame(Err(error)) => {
                self.fail(format!("live frame processing failed: {error:#}"));
            }
            CaptureReceiveOutcome::ProducerFailed(reason) => {
                self.fail(format!("live capture source failed: {reason}"));
            }
            CaptureReceiveOutcome::Cancelled => {
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
}
