//! Synchronous capture-to-localization pipeline over borrowed RGB8 frames.
//!
//! A caller constructs this pipeline before starting capture so its monotonic
//! clock origin predates the first frame, then invokes `process` from the sole
//! `CaptureFrameReceiver` callback. The camera-owning source remains on its
//! worker thread; inference borrows the receiver's bounded frame buffer and
//! completed scenes enter the existing newest-only loopback writer.

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use crate::calibration::ValidatedTableCalibrationV1;
use crate::capture::{
    CaptureFrameReceiver, CaptureFrameTiming, CaptureReceiveOutcome, CaptureTimestampSemantics,
    RgbFrameEnvelope,
};
use crate::capture_session_ack::CaptureSessionAcknowledgement;
use crate::detection_batch::{
    BatchOutcome, BatchTimestamp, BoundingBox, ClockKind as BatchClockKind, DetectionBatchV1,
    ImageSpace as BatchImageSpace, ModelIdentity, Point, SourceFrame as BatchSourceFrame,
    TimeUnit as BatchTimeUnit, TimestampStage as BatchTimestampStage,
};
use crate::geometry::{calibrate_detection_batch, AppliedCalibrationContext};
use crate::inference::{FrameInferenceEngine, FrameInferenceOutput, FrameInferenceTimings};
use crate::live_audit::{
    LiveAuditFrame, LiveAuditInferenceTimings, LiveAuditObservationOutcome, LiveAuditRecorder,
};
use crate::live_config::{CaptureSessionCompatibility, ExpectedCaptureSettings};
use crate::live_timing_spans::{FrameTimingContext, SpanGuard, SpanId};
use crate::public_state_lifecycle::{PublicStateDeliveryV1, SceneStateLifecycle};
use crate::scene_transport::{LatestSceneWriter, SceneTransportConfig, SceneTransportStats};
use crate::tracking::{SceneTrackerV1, TrackingConfigV1};
use crate::{
    encode_scene_snapshot_v1, ClockKind, SceneSnapshotV1, TimeUnit, Timestamp as SceneTimestamp,
    TimestampStage as SceneTimestampStage,
};

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// A detector that accepts the borrowed, packed RGB8 capture representation.
///
/// This trait keeps lifecycle and failure behavior testable without replacing
/// the production ONNX engine. Implementations must return a complete result
/// for one source observation or an error; an empty detection vector is a
/// successful empty observation.
pub trait Rgb8FrameDetector {
    fn infer_rgb8(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels: &[u8],
    ) -> Result<FrameInferenceOutput>;

    /// Run inference with optional nested audit timing. Implementations that
    /// do not expose internal stages retain their existing behavior; the live
    /// capture path still measures the detector as one inclusive span.
    fn infer_rgb8_with_timing(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels: &[u8],
        _timing: Option<(&FrameTimingContext, SpanId)>,
    ) -> Result<FrameInferenceOutput> {
        self.infer_rgb8(width_px, height_px, pixels)
    }

    fn model_identity(&self) -> &ModelIdentity;

    fn inference_settings(&self) -> &crate::detection_batch::InferenceSettings;
}

impl Rgb8FrameDetector for FrameInferenceEngine {
    fn infer_rgb8(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels: &[u8],
    ) -> Result<FrameInferenceOutput> {
        FrameInferenceEngine::infer_rgb8(self, width_px, height_px, pixels)
    }

    fn infer_rgb8_with_timing(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels: &[u8],
        timing: Option<(&FrameTimingContext, SpanId)>,
    ) -> Result<FrameInferenceOutput> {
        if let Some((context, parent)) = timing {
            FrameInferenceEngine::infer_rgb8_with_spans(
                self, width_px, height_px, pixels, context, parent,
            )
        } else {
            FrameInferenceEngine::infer_rgb8(self, width_px, height_px, pixels)
        }
    }

    fn model_identity(&self) -> &ModelIdentity {
        FrameInferenceEngine::model_identity(self)
    }

    fn inference_settings(&self) -> &crate::detection_batch::InferenceSettings {
        FrameInferenceEngine::inference_settings(self)
    }
}

/// Validated runtime choices for one capture-to-scene stream.
#[derive(Debug, Clone)]
pub struct CapturePipelineConfig {
    /// Run identity and process-local monotonic clock-domain identity.
    pub engine_instance_id: String,
    /// Effective capture/runtime configuration revision, compared with the
    /// expected settings when the pipeline is preflight-bound. This is
    /// distinct from the tracking subsystem's own configuration revision.
    pub capture_configuration_revision: u64,
    /// Must describe the bytes supplied by the selected capture adapter.
    pub image_space: BatchImageSpace,
    /// Maximum allowed host read-completion-to-publication age.
    pub maximum_frame_age: Duration,
    /// Calibration acknowledged by the runtime for this image geometry.
    /// When absent, scenes explicitly report table coordinates as unavailable.
    pub calibration: Option<AppliedCaptureCalibration>,
    /// Bounded newest-only loopback publication settings.
    pub scene_transport: SceneTransportConfig,
}

/// Validated calibration and its runtime-acknowledged application context.
///
/// The bundle's own camera profile and revision metadata must match the values
/// supplied by the control/runtime boundary. Per-frame resolution and image
/// space are checked again against each captured observation before any table
/// coordinate is published.
#[derive(Debug, Clone)]
pub struct AppliedCaptureCalibration {
    calibration: ValidatedTableCalibrationV1,
    calibration_revision: u64,
    camera_profile_id: String,
    rectification_revision: u64,
}

impl AppliedCaptureCalibration {
    pub fn new(
        calibration: ValidatedTableCalibrationV1,
        calibration_revision: u64,
        camera_profile_id: impl Into<String>,
        rectification_revision: u64,
    ) -> Result<Self> {
        let camera_profile_id = camera_profile_id.into();
        ensure!(
            calibration.calibration_revision() == calibration_revision,
            "applied calibration revision does not match the validated bundle"
        );
        ensure!(
            calibration.camera_profile_id() == camera_profile_id,
            "applied camera profile does not match the validated calibration bundle"
        );
        ensure!(
            calibration.rectification_revision() == rectification_revision,
            "applied rectification revision does not match the validated calibration bundle"
        );
        Ok(Self {
            calibration,
            calibration_revision,
            camera_profile_id,
            rectification_revision,
        })
    }
}

/// Result of synchronously processing one source frame.
#[derive(Debug, Clone, PartialEq)]
pub struct CapturePipelineResult {
    pub batch: DetectionBatchV1,
    pub scene: SceneSnapshotV1,
    pub association_capacity_evictions: usize,
    pub inference_timings: FrameInferenceTimings,
    /// Geometry time for an explicitly configured calibration. `None` means
    /// no table transform was configured for this runtime.
    pub calibration_ms: Option<f64>,
    /// Parent completion of the full frame-record read through inference start,
    /// including receiver queue wait. This is not sensor exposure latency.
    pub parent_record_read_complete_to_inference_start_ms: f64,
    /// Parent completion of the full frame-record read through successful
    /// insertion into the scene queue. Socket-write and receipt are separate.
    pub parent_record_read_complete_to_scene_queue_ms: f64,
}

/// An observation rejected before a scene can be published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSkipReason {
    StaleBeforeInference,
    StaleAfterInference,
    DuplicateSourceFrame,
    NonIncreasingSourceTime,
}

/// A frame rejected by an explicit validated-session binding before inference.
/// Device and session identifiers are provenance values, not credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureFrameRejection {
    SourceSessionMismatch {
        expected_session_id: String,
        actual_session_id: String,
    },
    FrameDimensionsMismatch {
        expected_width_px: u32,
        expected_height_px: u32,
        actual_width_px: u32,
        actual_height_px: u32,
    },
    TimestampSemanticsMismatch {
        expected: CaptureTimestampSemantics,
        actual: Option<CaptureTimestampSemantics>,
    },
}

/// Failure to bind a pipeline to one validated source session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapturePipelineBindError {
    FramesAlreadyProcessed,
    AlreadyBound,
    ImageSpaceMismatch,
    MaximumFrameAgeMismatch,
    ConfigurationRevisionMismatch,
    CameraProfileMismatch,
    RectificationRevisionMismatch,
    CalibrationMismatch(&'static str),
    LiveHandshakeFailed(String),
}

impl std::fmt::Display for CapturePipelineBindError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "capture pipeline preflight bind failed: {self:?}"
        )
    }
}

impl std::error::Error for CapturePipelineBindError {}

#[derive(Debug, Clone, PartialEq)]
pub enum CapturePipelineOutcome {
    Published(Box<CapturePipelineResult>),
    Skipped(CaptureSkipReason),
    Rejected(CaptureFrameRejection),
}

/// Counters for frame processing, excluding capture-adapter and receiver loss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CapturePipelineStats {
    pub frames_received: u64,
    /// Frames rejected by the optional preflight binding before inference.
    pub preflight_rejections: u64,
    pub inference_started: u64,
    pub inference_failures: u64,
    pub scene_processing_failures: u64,
    pub publication_failures: u64,
    /// Successfully published frames that evicted old association history.
    pub association_capacity_pressure_frames: u64,
    /// Number of old associations evicted while preserving current locations.
    pub association_capacity_evictions: u64,
    pub stale_frames: u64,
    pub duplicate_frames: u64,
    pub out_of_order_frames: u64,
    pub scenes_submitted: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceOrderKey {
    clock: BatchClockKind,
    domain_id: String,
    ticks: u64,
}

/// Runs real per-frame inference, tracking, and bounded scene submission.
///
/// Construct this before starting the capture worker. Its clock origin is
/// process-local and remains distinct from any source-supplied clock. The
/// source timestamp is copied unchanged when present; when absent, the
/// captured host read-completion `Instant` is represented as a monotonic
/// `CameraReadComplete` timestamp in this run's domain. That fallback is never
/// labeled as sensor exposure.
pub struct CaptureScenePipeline<D: Rgb8FrameDetector> {
    detector: D,
    tracker: SceneTrackerV1,
    writer: LatestSceneWriter,
    /// Optional side-channel recorder; `None` creates no audit worker or queue.
    audit: Option<LiveAuditRecorder>,
    config: CapturePipelineConfig,
    /// Present only for live producer configurations whose gateway consumes
    /// Rust-owned public-state events. Replay retains the raw scene v1 wire.
    state_lifecycle: Option<SceneStateLifecycle>,
    state_lifecycle_session_id: Option<String>,
    /// Compatibility with the opened source session. This does not prove
    /// application of settings absent from the source report.
    preflight_binding: Option<CaptureSessionCompatibility>,
    /// Strict wire-parsed live settings require a successful v3 receiver
    /// handshake before any frame can enter inference.
    live_startup_required: bool,
    tracking_camera_profile_id: String,
    tracking_rectification_revision: u64,
    tracking_configuration_revision: u64,
    clock_origin: Instant,
    next_batch_sequence: u64,
    current_session_id: Option<String>,
    recent_frame_ids: VecDeque<String>,
    recent_frame_id_set: HashSet<String>,
    recent_frame_capacity: usize,
    last_source_order: Option<SourceOrderKey>,
    stats: CapturePipelineStats,
}

impl<D: Rgb8FrameDetector> CaptureScenePipeline<D> {
    /// Create the pipeline and establish its monotonic origin.
    ///
    /// Call this before `CaptureRuntimeCoordinator::start`. The validated model,
    /// tracking, and writer configuration bounds the processing and transport
    /// resources used by this stage.
    pub fn new(
        detector: D,
        tracking: TrackingConfigV1,
        config: CapturePipelineConfig,
    ) -> Result<Self> {
        Self::new_with_audit(detector, tracking, config, None)
    }

    /// Create a pipeline with a separately bounded audit recorder. This
    /// additive constructor leaves existing callers and the scene schema
    /// unchanged. The recorder only receives scalar timings and identities.
    pub fn new_with_audit(
        detector: D,
        tracking: TrackingConfigV1,
        config: CapturePipelineConfig,
        audit: Option<LiveAuditRecorder>,
    ) -> Result<Self> {
        let maximum_per_frame_detections = detector.inference_settings().max_detections as usize;
        ensure!(
            tracking.maximum_active_tracks() >= maximum_per_frame_detections,
            "maximum active tracks ({}) must cover the detector's configured per-frame detection maximum ({maximum_per_frame_detections})",
            tracking.maximum_active_tracks()
        );
        validate_config(&config, &tracking)?;
        let recent_frame_capacity = tracking.recent_observation_capacity();
        let tracking_camera_profile_id = tracking.camera_profile_id().to_owned();
        let tracking_configuration_revision = tracking.configuration_revision();
        let tracking_rectification_revision = tracking.rectification_revision();
        let state_lifecycle = config
            .scene_transport
            .public_state_delivery_enabled
            .then(SceneStateLifecycle::default);
        let tracker = SceneTrackerV1::new(tracking);
        let writer = LatestSceneWriter::start_with_audit(config.scene_transport, audit.clone())?;
        Ok(Self {
            detector,
            tracker,
            writer,
            audit,
            config,
            state_lifecycle,
            state_lifecycle_session_id: None,
            preflight_binding: None,
            live_startup_required: false,
            tracking_camera_profile_id,
            tracking_configuration_revision,
            tracking_rectification_revision,
            clock_origin: Instant::now(),
            next_batch_sequence: 1,
            current_session_id: None,
            recent_frame_ids: VecDeque::with_capacity(recent_frame_capacity),
            recent_frame_id_set: HashSet::with_capacity(recent_frame_capacity),
            recent_frame_capacity,
            last_source_order: None,
            stats: CapturePipelineStats::default(),
        })
    }

    /// Bind this already-constructed pipeline to the compatibility returned
    /// after a capture source opens. Constructing first preserves the
    /// pipeline's monotonic clock origin. Binding must happen before the first
    /// frame is consumed and may happen only once.
    ///
    /// This compares expected settings with the pipeline's effective local
    /// configuration. Preflight also checks the source-reported geometry and
    /// required control readbacks before the first source read. These reports
    /// are not physical-device proof. Calibration remains pipeline-owned and
    /// is checked separately; missing runtime-applied calibration is allowed,
    /// in which case table coordinates remain unavailable. These checks bind
    /// provenance and configuration; identifiers are not credentials and no
    /// authentication is performed.
    pub fn bind_preflight(
        &mut self,
        compatibility: CaptureSessionCompatibility,
    ) -> Result<(), CapturePipelineBindError> {
        if self.preflight_binding.is_some() {
            return Err(CapturePipelineBindError::AlreadyBound);
        }
        if self.stats.frames_received > 0 {
            return Err(CapturePipelineBindError::FramesAlreadyProcessed);
        }

        if compatibility.request_sha256().is_some() {
            self.live_startup_required = true;
        } else if self.live_startup_required {
            return Err(CapturePipelineBindError::LiveHandshakeFailed(
                "strict live startup cannot fall back to replay protocol v2".to_owned(),
            ));
        }

        let expected = compatibility.expected_settings();
        self.validate_expected_capture_settings(expected)?;

        // Strict wire-parsed settings carry the exact request digest. In that
        // case live startup must receive a matching v3 response before the
        // caller is allowed to begin camera reads. Hand-built expectations
        // retain the existing v2 behavior used by replay and isolated tests.
        if compatibility.request_sha256().is_some() {
            let acknowledgement = CaptureSessionAcknowledgement::from_compatible_session(
                &compatibility,
            )
            .map_err(|error| CapturePipelineBindError::LiveHandshakeFailed(format!("{error:#}")))?;
            if let Err(error) = self
                .writer
                .bind_live_session(&self.config.engine_instance_id, acknowledgement)
            {
                return Err(CapturePipelineBindError::LiveHandshakeFailed(format!(
                    "{error:#}"
                )));
            }
        }

        self.preflight_binding = Some(compatibility);
        Ok(())
    }

    /// Check the duplicated capture-facing settings before a source is opened.
    ///
    /// A live runtime must derive its frame budgets from the parsed request and
    /// reject stale pipeline/tracker configuration before acquiring a camera.
    /// This method intentionally does not establish source compatibility or
    /// start the request-bound network handshake; those require the opened
    /// source report and remain the responsibility of `bind_preflight`.
    pub fn validate_expected_capture_settings(
        &self,
        expected: &ExpectedCaptureSettings,
    ) -> Result<(), CapturePipelineBindError> {
        let expected_image_space = match expected.image_space() {
            crate::ImageSpace::Raw => BatchImageSpace::Raw,
            crate::ImageSpace::Rectified => BatchImageSpace::Rectified,
        };
        if self.config.image_space != expected_image_space {
            return Err(CapturePipelineBindError::ImageSpaceMismatch);
        }
        if self.config.maximum_frame_age != expected.maximum_frame_age() {
            return Err(CapturePipelineBindError::MaximumFrameAgeMismatch);
        }
        if self.config.capture_configuration_revision != expected.configuration_revision() {
            return Err(CapturePipelineBindError::ConfigurationRevisionMismatch);
        }
        if self.tracking_camera_profile_id != expected.camera_profile_id() {
            return Err(CapturePipelineBindError::CameraProfileMismatch);
        }
        if self.tracking_rectification_revision != expected.rectification_revision() {
            return Err(CapturePipelineBindError::RectificationRevisionMismatch);
        }

        if let Some(applied) = &self.config.calibration {
            let Some(expected_calibration) = expected.calibration() else {
                return Err(CapturePipelineBindError::CalibrationMismatch(
                    "runtime calibration is configured but no expected calibration bundle exists",
                ));
            };
            if applied.calibration_revision != expected.calibration_revision()
                || applied.calibration_revision != expected_calibration.calibration_revision()
            {
                return Err(CapturePipelineBindError::CalibrationMismatch(
                    "runtime calibration revision differs from the expected revision",
                ));
            }
            if applied.camera_profile_id != expected.camera_profile_id()
                || applied.camera_profile_id != expected_calibration.camera_profile_id()
            {
                return Err(CapturePipelineBindError::CalibrationMismatch(
                    "runtime calibration profile differs from the expected profile",
                ));
            }
            if applied.rectification_revision != expected.rectification_revision()
                || applied.rectification_revision != expected_calibration.rectification_revision()
            {
                return Err(CapturePipelineBindError::CalibrationMismatch(
                    "runtime calibration rectification differs from the expected revision",
                ));
            }
            if applied.calibration.image_space() != expected.image_space()
                || applied.calibration.image_space() != expected_calibration.image_space()
            {
                return Err(CapturePipelineBindError::CalibrationMismatch(
                    "runtime calibration image space differs from the expected image space",
                ));
            }
            if applied.calibration.source_dimensions_px()
                != expected_calibration.source_dimensions_px()
            {
                return Err(CapturePipelineBindError::CalibrationMismatch(
                    "runtime calibration dimensions differ from the expected bundle",
                ));
            }
            if &applied.calibration != expected_calibration {
                return Err(CapturePipelineBindError::CalibrationMismatch(
                    "runtime calibration bundle differs from the expected bundle",
                ));
            }
        }
        Ok(())
    }

    /// Process a borrowed capture frame and enqueue one complete scene.
    ///
    /// Duplicate and stale observations are rejected before inference where
    /// their provenance permits it. A scene is committed to tracker state only
    /// after it validates and enters the bounded writer slot. The writer's
    /// queue replaces older pending scenes and never waits for socket I/O.
    pub fn process(&mut self, frame: &RgbFrameEnvelope) -> Result<CapturePipelineOutcome> {
        self.process_with_receiver_queue_wait(frame, None)
    }

    fn process_with_receiver_queue_wait(
        &mut self,
        frame: &RgbFrameEnvelope,
        receiver_queue_wait_ns: Option<u64>,
    ) -> Result<CapturePipelineOutcome> {
        let capture_timing = receiver_queue_wait_ns
            .map(|queue_wait_ns| {
                let mut timing = frame.capture_timing();
                timing.parent_receiver_queue_wait_ns = Some(queue_wait_ns);
                timing
            })
            .unwrap_or_else(|| frame.capture_timing());
        if self.live_startup_required && self.preflight_binding.is_none() {
            self.audit_failure(
                frame,
                None,
                "session_readiness",
                "missing_preflight_binding",
            );
            anyhow::bail!(
                "live capture frame cannot be processed before a successful request-bound receiver handshake"
            );
        }
        self.stats.frames_received = self.stats.frames_received.saturating_add(1);
        if let Some(rejection) = self.preflight_frame_rejection(frame) {
            self.stats.preflight_rejections = self.stats.preflight_rejections.saturating_add(1);
            self.audit_rejection(frame, &rejection);
            return Ok(CapturePipelineOutcome::Rejected(rejection));
        }
        let age = match frame_age(frame.host_read_completed_at(), Instant::now()) {
            Ok(age) => age,
            Err(error) => {
                self.audit_failure(frame, None, "freshness_check", "clock_error");
                return Err(error);
            }
        };
        if age > self.config.maximum_frame_age {
            self.stats.stale_frames = self.stats.stale_frames.saturating_add(1);
            self.audit_skip(frame, None, CaptureSkipReason::StaleBeforeInference);
            return Ok(CapturePipelineOutcome::Skipped(
                CaptureSkipReason::StaleBeforeInference,
            ));
        }

        let effective_source_timestamp = match self.source_timestamp(frame) {
            Ok(timestamp) => timestamp,
            Err(error) => {
                self.audit_failure(
                    frame,
                    None,
                    "source_timestamp",
                    "timestamp_conversion_failed",
                );
                return Err(error);
            }
        };
        let admission = match self.admit_source_observation(frame, &effective_source_timestamp) {
            Ok(admission) => admission,
            Err(error) => {
                self.audit_failure(frame, None, "observation_admission", "admission_failed");
                return Err(error);
            }
        };
        if let Some(reason) = admission {
            match reason {
                CaptureSkipReason::DuplicateSourceFrame => {
                    self.stats.duplicate_frames = self.stats.duplicate_frames.saturating_add(1);
                }
                CaptureSkipReason::NonIncreasingSourceTime => {
                    self.stats.out_of_order_frames =
                        self.stats.out_of_order_frames.saturating_add(1);
                }
                CaptureSkipReason::StaleBeforeInference
                | CaptureSkipReason::StaleAfterInference => {
                    self.stats.stale_frames = self.stats.stale_frames.saturating_add(1);
                }
            }
            self.audit_skip(frame, None, reason);
            return Ok(CapturePipelineOutcome::Skipped(reason));
        }

        // Construct timing state only for explicitly audited frames. The
        // disabled production path creates no span context, mutex, or span
        // buffer. The context starts at host read-completion, never camera
        // exposure, so its elapsed value remains in the verified parent clock.
        let timing_context = self
            .audit
            .as_ref()
            .map(|_| FrameTimingContext::new(frame.host_read_completed_at()));
        let mut frame_span = timing_context.as_ref().and_then(|timing| {
            timing
                .start_span_at("frame_processing", None, frame.host_read_completed_at())
                .ok()
        });
        let frame_span_id = frame_span.as_ref().map(SpanGuard::id);
        if let (Some(timing), Some(parent), Some(queue_wait_ns)) = (
            timing_context.as_ref(),
            frame_span_id,
            receiver_queue_wait_ns,
        ) {
            if let (Some(queue_wait_end), Ok(queue_wait_span)) = (
                frame
                    .host_read_completed_at()
                    .checked_add(Duration::from_nanos(queue_wait_ns)),
                timing.start_span_at(
                    "capture_receiver_queue_wait",
                    Some(parent),
                    frame.host_read_completed_at(),
                ),
            ) {
                let _ = queue_wait_span.finish_at(queue_wait_end);
            }
        }

        let source = frame.source();
        let inference_started = Instant::now();
        let parent_record_read_complete_to_inference_start_ms =
            match frame_age(frame.host_read_completed_at(), inference_started) {
                Ok(age) => age.as_secs_f64() * 1_000.0,
                Err(error) => {
                    self.audit_failure(frame, None, "inference_scheduling", "clock_error");
                    return Err(error);
                }
            };
        self.stats.inference_started = self.stats.inference_started.saturating_add(1);
        let detector_span =
            start_timing_child_span(timing_context.as_ref(), "detector", frame_span_id);
        let inference_timing_parent = detector_span.as_ref().map(SpanGuard::id);
        let inference = match self.detector.infer_rgb8_with_timing(
            source.width_px,
            source.height_px,
            frame.rgb8(),
            timing_context.as_ref().zip(inference_timing_parent),
        ) {
            Ok(output) => output,
            Err(error) => {
                self.stats.inference_failures = self.stats.inference_failures.saturating_add(1);
                self.audit_failure(frame, None, "inference", "inference_failed");
                return Err(error).context("RGB8 inference failed for captured frame");
            }
        };
        drop(detector_span);
        let inference_completed = Instant::now();
        if inference.source_width_px != source.width_px
            || inference.source_height_px != source.height_px
        {
            self.stats.inference_failures = self.stats.inference_failures.saturating_add(1);
            self.audit_failure(
                frame,
                None,
                "inference_output_validation",
                "source_dimensions_mismatch",
            );
            anyhow::bail!("inference result dimensions disagree with captured source frame");
        }
        let age = match frame_age(frame.host_read_completed_at(), inference_completed) {
            Ok(age) => age,
            Err(error) => {
                self.audit_failure(frame, None, "freshness_check", "clock_error");
                return Err(error);
            }
        };
        if age > self.config.maximum_frame_age {
            self.stats.stale_frames = self.stats.stale_frames.saturating_add(1);
            self.audit_stale_after_inference(
                frame,
                capture_timing,
                inference.timings,
                parent_record_read_complete_to_inference_start_ms,
                age,
            );
            return Ok(CapturePipelineOutcome::Skipped(
                CaptureSkipReason::StaleAfterInference,
            ));
        }

        let batch = match self.build_batch(
            frame,
            &inference,
            effective_source_timestamp,
            inference_started,
            inference_completed,
        ) {
            Ok(batch) => batch,
            Err(error) => {
                self.stats.scene_processing_failures =
                    self.stats.scene_processing_failures.saturating_add(1);
                self.audit_failure(frame, None, "batch_assembly", "batch_assembly_failed");
                return Err(error).context("failed to build captured-frame detection batch");
            }
        };

        let published_at = match scene_publication_timestamp(
            self.clock_origin,
            &self.config.engine_instance_id,
            Instant::now(),
        ) {
            Ok(timestamp) => timestamp,
            Err(error) => {
                self.stats.scene_processing_failures =
                    self.stats.scene_processing_failures.saturating_add(1);
                self.audit_failure(
                    frame,
                    None,
                    "publication_timestamp",
                    "publication_timestamp_failed",
                );
                return Err(error).context("failed to timestamp captured scene publication");
            }
        };
        let calibration_started = Instant::now();
        let calibration_span =
            start_timing_child_span(timing_context.as_ref(), "calibration", frame_span_id);
        let calibrated = self.config.calibration.as_ref().map(|applied| {
            let context = AppliedCalibrationContext {
                calibration_revision: applied.calibration_revision,
                camera_profile_id: &applied.camera_profile_id,
                rectification_revision: applied.rectification_revision,
            };
            (
                applied.calibration_revision,
                calibrate_detection_batch(&batch, &applied.calibration, context),
            )
        });
        drop(calibration_span);
        let calibration_ms = self
            .config
            .calibration
            .as_ref()
            .map(|_| calibration_started.elapsed().as_secs_f64() * 1_000.0);
        let tracking_started = self.audit.as_ref().map(|_| Instant::now());
        let tracking_span = start_timing_child_span(
            timing_context.as_ref(),
            "tracking_and_state_assembly",
            frame_span_id,
        );
        let mut candidate_tracker = self.tracker.clone();
        let tracked_scene = match candidate_tracker.process_with_precalibrated_batch_report(
            &batch,
            published_at,
            calibrated,
        ) {
            Ok(tracked_scene) => tracked_scene,
            Err(error) => {
                self.stats.scene_processing_failures =
                    self.stats.scene_processing_failures.saturating_add(1);
                self.audit_failure(
                    frame,
                    None,
                    "tracking_and_state_assembly",
                    "state_assembly_failed",
                );
                return Err(error).context("failed to build complete captured scene");
            }
        };
        drop(tracking_span);
        let association_capacity_evictions = tracked_scene.association_capacity_evictions;
        let scene = tracked_scene.snapshot;
        let mut tracking_state_ms =
            tracking_started.map(|started| started.elapsed().as_secs_f64() * 1_000.0);
        let serialization_started = self.audit.as_ref().map(|_| Instant::now());
        let scene_serialization_span = start_timing_child_span(
            timing_context.as_ref(),
            "scene_json_serialization",
            frame_span_id,
        );
        let raw_payload = match encode_scene_snapshot_v1(&scene) {
            Ok(payload) => payload,
            Err(error) => {
                self.stats.scene_processing_failures =
                    self.stats.scene_processing_failures.saturating_add(1);
                self.audit_failure(
                    frame,
                    Some(scene.transport_sequence),
                    "scene_serialization",
                    "scene_serialization_failed",
                );
                return Err(error).context("captured scene failed schema validation");
            }
        };
        drop(scene_serialization_span);
        let mut scene_serialization_ms =
            serialization_started.map(|started| started.elapsed().as_secs_f64() * 1_000.0);
        let age = match frame_age(frame.host_read_completed_at(), Instant::now()) {
            Ok(age) => age,
            Err(error) => {
                self.audit_failure(
                    frame,
                    Some(scene.transport_sequence),
                    "freshness_check",
                    "clock_error",
                );
                return Err(error);
            }
        };
        if age > self.config.maximum_frame_age {
            self.stats.stale_frames = self.stats.stale_frames.saturating_add(1);
            self.audit_skip(
                frame,
                Some(scene.transport_sequence),
                CaptureSkipReason::StaleAfterInference,
            );
            return Ok(CapturePipelineOutcome::Skipped(
                CaptureSkipReason::StaleAfterInference,
            ));
        }

        let lifecycle_span = self.state_lifecycle.as_ref().and_then(|_| {
            start_timing_child_span(
                timing_context.as_ref(),
                "public_state_lifecycle",
                frame_span_id,
            )
        });
        let payload = if let Some(lifecycle) = self.state_lifecycle.as_mut() {
            let state_assembly_started = self.audit.as_ref().map(|_| Instant::now());
            let session_id = &scene.source.session_id;
            if self.state_lifecycle_session_id.as_deref() != Some(session_id) {
                lifecycle
                    .activate_source(&scene.engine_instance_id, session_id)
                    .context("failed to activate Rust public-state producer epoch")?;
                self.state_lifecycle_session_id = Some(session_id.clone());
            }
            let received_ns =
                monotonic_elapsed_ns(self.clock_origin, frame.host_read_completed_at())?;
            let now = Instant::now();
            let now_ns = monotonic_elapsed_ns(self.clock_origin, now)?;
            lifecycle
                .ingest(&scene, received_ns, now_ns, None, None)
                .context("Rust public-state lifecycle rejected a captured scene")?;
            let current_state = lifecycle
                .current(now_ns)
                .context("failed to read current Rust public state")?;
            if let Some(started) = state_assembly_started {
                if let Some(stage_ms) = tracking_state_ms.as_mut() {
                    *stage_ms += started.elapsed().as_secs_f64() * 1_000.0;
                }
            }
            let envelope_serialization_started = self.audit.as_ref().map(|_| Instant::now());
            let event = PublicStateDeliveryV1::new(&scene, current_state.as_ref());
            let delivery_serialization_span = start_timing_child_span(
                timing_context.as_ref(),
                "public_state_json_serialization",
                frame_span_id,
            );
            let event_payload = serde_json::to_vec(&event)
                .context("failed to serialize Rust public-state delivery v1")?;
            drop(delivery_serialization_span);
            if let (Some(total_ms), Some(started)) = (
                scene_serialization_ms.as_mut(),
                envelope_serialization_started,
            ) {
                *total_ms += started.elapsed().as_secs_f64() * 1_000.0;
            }
            event_payload
        } else {
            raw_payload
        };
        drop(lifecycle_span);

        let next_batch_sequence = match self.next_batch_sequence.checked_add(1) {
            Some(sequence) => sequence,
            None => {
                self.audit_failure(
                    frame,
                    Some(scene.transport_sequence),
                    "sequence_assignment",
                    "sequence_overflow",
                );
                anyhow::bail!("captured batch sequence overflowed");
            }
        };
        let queue_submit_started = self.audit.as_ref().map(|_| Instant::now());
        let queue_submit_span =
            start_timing_child_span(timing_context.as_ref(), "scene_queue_submit", frame_span_id);
        let freshness_deadline = frame
            .host_read_completed_at()
            .checked_add(self.config.maximum_frame_age)
            .context("captured frame freshness deadline cannot be represented")?;
        if let Err(error) = self.writer.submit_live_with_frame_id_and_deadline(
            scene.transport_sequence,
            &scene.engine_instance_id,
            &scene.source.session_id,
            &scene.source.frame_id,
            freshness_deadline,
            payload,
        ) {
            self.stats.publication_failures = self.stats.publication_failures.saturating_add(1);
            self.audit_failure(
                frame,
                Some(scene.transport_sequence),
                "scene_queue_submit",
                "scene_queue_submit_failed",
            );
            return Err(error).context("failed to queue captured scene snapshot");
        }
        drop(queue_submit_span);

        self.tracker = candidate_tracker;
        self.next_batch_sequence = next_batch_sequence;
        self.stats.scenes_submitted = self.stats.scenes_submitted.saturating_add(1);
        if association_capacity_evictions > 0 {
            self.stats.association_capacity_pressure_frames = self
                .stats
                .association_capacity_pressure_frames
                .saturating_add(1);
            self.stats.association_capacity_evictions = self
                .stats
                .association_capacity_evictions
                .saturating_add(u64::try_from(association_capacity_evictions).unwrap_or(u64::MAX));
        }
        if let Some(frame_span) = frame_span.take() {
            let _ = frame_span.finish();
        }
        let frame_timing = timing_context
            .as_ref()
            .and_then(|timing| timing.snapshot().ok());
        let parent_record_read_complete_to_scene_queue_ms =
            match frame_age(frame.host_read_completed_at(), Instant::now()) {
                Ok(age) => age.as_secs_f64() * 1_000.0,
                Err(error) => {
                    self.audit_failure(
                        frame,
                        Some(scene.transport_sequence),
                        "post_submission_age",
                        "clock_error",
                    );
                    return Err(error);
                }
            };
        let scene_queue_submit_ms =
            queue_submit_started.map(|started| started.elapsed().as_secs_f64() * 1_000.0);
        if let Some(audit) = self.audit.as_ref() {
            if association_capacity_evictions > 0 {
                let source = frame.source();
                let reason = format!(
                    "association_capacity_pressure_evicted_{association_capacity_evictions}"
                );
                audit.record_observation_outcome(LiveAuditObservationOutcome {
                    engine_instance_id: &scene.engine_instance_id,
                    source_session_id: &scene.source.session_id,
                    frame_id: &scene.source.frame_id,
                    transport_sequence: Some(scene.transport_sequence),
                    configuration_revision: self.config.capture_configuration_revision,
                    tracking_configuration_revision: self.tracking_configuration_revision,
                    rectification_revision: self.tracking_rectification_revision,
                    calibration_revision: self
                        .config
                        .calibration
                        .as_ref()
                        .map(|calibration| calibration.calibration_revision),
                    source_timestamp: source.timestamp.as_ref(),
                    outcome: "degraded",
                    stage: "track_association",
                    reason: &reason,
                    inference_timings: None,
                });
            }
            audit.record_frame(LiveAuditFrame {
                engine_instance_id: &scene.engine_instance_id,
                source_session_id: &scene.source.session_id,
                frame_id: &scene.source.frame_id,
                transport_sequence: scene.transport_sequence,
                configuration_revision: self.config.capture_configuration_revision,
                tracking_configuration_revision: self.tracking_configuration_revision,
                rectification_revision: self.tracking_rectification_revision,
                calibration_revision: self
                    .config
                    .calibration
                    .as_ref()
                    .map(|calibration| calibration.calibration_revision),
                detection_count: batch.detections.len(),
                track_count: scene.balls.len(),
                source_timestamp: batch.source.timestamp.as_ref(),
                capture_timing,
                parent_record_read_complete_to_inference_start_ms,
                preprocess_ms: inference.timings.preprocess_ms,
                inference_ms: inference.timings.inference_ms,
                decode_and_nms_ms: inference.timings.decode_and_nms_ms,
                engine_total_ms: inference.timings.engine_total_ms,
                calibration_ms,
                tracking_state_ms,
                scene_serialization_ms,
                scene_queue_submit_ms,
                parent_record_read_complete_to_scene_queue_ms,
                frame_timing: frame_timing.as_ref(),
            });
        }
        Ok(CapturePipelineOutcome::Published(Box::new(
            CapturePipelineResult {
                batch,
                scene,
                association_capacity_evictions,
                inference_timings: inference.timings,
                calibration_ms,
                parent_record_read_complete_to_inference_start_ms,
                parent_record_read_complete_to_scene_queue_ms,
            },
        )))
    }

    /// Receive one frame and process it inside the receiver's synchronous
    /// borrow callback. This keeps the capture-to-inference ownership boundary
    /// explicit while letting the caller handle timeout, stop, cancellation,
    /// and producer failure.
    pub fn receive_one(
        &mut self,
        receiver: &mut CaptureFrameReceiver,
        timeout: Duration,
    ) -> CaptureReceiveOutcome<Result<CapturePipelineOutcome>> {
        let measure_queue_wait = self.audit.is_some();
        receiver.receive_with_timeout_and_queue_timing(
            timeout,
            measure_queue_wait,
            |frame, queue_wait_ns| self.process_with_receiver_queue_wait(frame, queue_wait_ns),
        )
    }

    /// Current local processing counters.
    pub fn stats(&self) -> CapturePipelineStats {
        self.stats
    }

    /// Current bounded scene-writer counters.
    pub fn transport_stats(&self) -> SceneTransportStats {
        self.writer.stats()
    }

    /// Stop the writer after its current bounded attempt and return counters.
    pub fn shutdown(&mut self) -> SceneTransportStats {
        self.writer.shutdown()
    }

    fn source_timestamp(&self, frame: &RgbFrameEnvelope) -> Result<BatchTimestamp> {
        match frame.source().timestamp.as_ref() {
            Some(timestamp) => Ok(timestamp.clone()),
            None => batch_monotonic_timestamp(
                self.clock_origin,
                frame.host_read_completed_at(),
                &self.config.engine_instance_id,
                BatchTimestampStage::CameraReadComplete,
            ),
        }
    }

    fn audit_skip(
        &self,
        frame: &RgbFrameEnvelope,
        transport_sequence: Option<u64>,
        reason: CaptureSkipReason,
    ) {
        let (stage, reason) = match reason {
            CaptureSkipReason::StaleBeforeInference => {
                ("freshness_check", "stale_before_inference")
            }
            CaptureSkipReason::StaleAfterInference => ("freshness_check", "stale_after_inference"),
            CaptureSkipReason::DuplicateSourceFrame => {
                ("observation_admission", "duplicate_source_frame")
            }
            CaptureSkipReason::NonIncreasingSourceTime => {
                ("observation_admission", "non_increasing_source_time")
            }
        };
        self.audit_observation(frame, transport_sequence, "skipped", stage, reason);
    }

    fn audit_stale_after_inference(
        &self,
        frame: &RgbFrameEnvelope,
        capture_timing: CaptureFrameTiming,
        inference: FrameInferenceTimings,
        parent_record_read_complete_to_inference_start_ms: f64,
        age: Duration,
    ) {
        self.audit_observation_with_inference_timings(
            frame,
            None,
            "skipped",
            "freshness_check",
            "stale_after_inference",
            Some(LiveAuditInferenceTimings {
                capture_timing,
                parent_record_read_complete_to_inference_start_ms,
                inference,
                parent_record_read_complete_to_inference_complete_ms: age.as_secs_f64() * 1_000.0,
                maximum_frame_age_ms: self.config.maximum_frame_age.as_secs_f64() * 1_000.0,
            }),
        );
    }

    fn audit_rejection(&self, frame: &RgbFrameEnvelope, rejection: &CaptureFrameRejection) {
        let reason = match rejection {
            CaptureFrameRejection::SourceSessionMismatch { .. } => "source_session_mismatch",
            CaptureFrameRejection::FrameDimensionsMismatch { .. } => "frame_dimensions_mismatch",
            CaptureFrameRejection::TimestampSemanticsMismatch { .. } => {
                "timestamp_semantics_mismatch"
            }
        };
        self.audit_observation(frame, None, "rejected", "preflight", reason);
    }

    fn audit_failure(
        &self,
        frame: &RgbFrameEnvelope,
        transport_sequence: Option<u64>,
        stage: &'static str,
        reason: &'static str,
    ) {
        self.audit_observation(frame, transport_sequence, "failed", stage, reason);
    }

    fn audit_observation(
        &self,
        frame: &RgbFrameEnvelope,
        transport_sequence: Option<u64>,
        outcome: &'static str,
        stage: &'static str,
        reason: &'static str,
    ) {
        self.audit_observation_with_inference_timings(
            frame,
            transport_sequence,
            outcome,
            stage,
            reason,
            None,
        );
    }

    fn audit_observation_with_inference_timings(
        &self,
        frame: &RgbFrameEnvelope,
        transport_sequence: Option<u64>,
        outcome: &'static str,
        stage: &'static str,
        reason: &'static str,
        inference_timings: Option<LiveAuditInferenceTimings>,
    ) {
        let Some(audit) = self.audit.as_ref() else {
            return;
        };
        let source = frame.source();
        audit.record_observation_outcome(LiveAuditObservationOutcome {
            engine_instance_id: &self.config.engine_instance_id,
            source_session_id: &source.session_id,
            frame_id: &source.frame_id,
            transport_sequence,
            configuration_revision: self.config.capture_configuration_revision,
            tracking_configuration_revision: self.tracking_configuration_revision,
            rectification_revision: self.tracking_rectification_revision,
            calibration_revision: self
                .config
                .calibration
                .as_ref()
                .map(|calibration| calibration.calibration_revision),
            source_timestamp: source.timestamp.as_ref(),
            outcome,
            stage,
            reason,
            inference_timings,
        });
    }

    fn preflight_frame_rejection(&self, frame: &RgbFrameEnvelope) -> Option<CaptureFrameRejection> {
        let binding = self.preflight_binding.as_ref()?;
        let source = frame.source();
        if source.session_id != binding.session_id() {
            return Some(CaptureFrameRejection::SourceSessionMismatch {
                expected_session_id: binding.session_id().to_owned(),
                actual_session_id: source.session_id.clone(),
            });
        }

        let output_geometry = binding.output_frame_geometry();
        if source.width_px != output_geometry.width_px()
            || source.height_px != output_geometry.height_px()
        {
            return Some(CaptureFrameRejection::FrameDimensionsMismatch {
                expected_width_px: output_geometry.width_px(),
                expected_height_px: output_geometry.height_px(),
                actual_width_px: source.width_px,
                actual_height_px: source.height_px,
            });
        }

        let expected = binding.timestamp_semantics();
        let actual =
            source
                .timestamp
                .as_ref()
                .map(|timestamp| CaptureTimestampSemantics::SourceTimestamp {
                    unit: timestamp.unit,
                    clock: timestamp.clock,
                    domain_id: timestamp.domain_id.clone(),
                    stage: timestamp.stage,
                });
        let timestamp_matches = match (expected, actual.as_ref()) {
            (CaptureTimestampSemantics::ReadCompletionOnly, None) => true,
            (
                CaptureTimestampSemantics::SourceTimestamp { .. },
                Some(CaptureTimestampSemantics::SourceTimestamp { .. }),
            ) => actual.as_ref() == Some(expected),
            _ => false,
        };
        (!timestamp_matches).then(|| CaptureFrameRejection::TimestampSemanticsMismatch {
            expected: expected.clone(),
            actual,
        })
    }

    fn admit_source_observation(
        &mut self,
        frame: &RgbFrameEnvelope,
        timestamp: &BatchTimestamp,
    ) -> Result<Option<CaptureSkipReason>> {
        let source = frame.source();
        if self.current_session_id.as_deref() != Some(source.session_id.as_str()) {
            self.current_session_id = Some(source.session_id.clone());
            self.recent_frame_ids.clear();
            self.recent_frame_id_set.clear();
            self.last_source_order = None;
        }
        if self.recent_frame_id_set.contains(&source.frame_id) {
            return Ok(Some(CaptureSkipReason::DuplicateSourceFrame));
        }
        let ticks = timestamp
            .ticks
            .parse::<u64>()
            .context("source timestamp ticks are not an unsigned 64-bit count")?;
        if self.last_source_order.as_ref().is_some_and(|previous| {
            previous.clock == timestamp.clock
                && previous.domain_id == timestamp.domain_id
                && ticks <= previous.ticks
        }) {
            return Ok(Some(CaptureSkipReason::NonIncreasingSourceTime));
        }

        self.last_source_order = Some(SourceOrderKey {
            clock: timestamp.clock,
            domain_id: timestamp.domain_id.clone(),
            ticks,
        });
        self.recent_frame_ids.push_back(source.frame_id.clone());
        self.recent_frame_id_set.insert(source.frame_id.clone());
        while self.recent_frame_ids.len() > self.recent_frame_capacity {
            if let Some(expired) = self.recent_frame_ids.pop_front() {
                self.recent_frame_id_set.remove(&expired);
            }
        }
        Ok(None)
    }

    fn build_batch(
        &self,
        frame: &RgbFrameEnvelope,
        inference: &FrameInferenceOutput,
        source_timestamp: BatchTimestamp,
        inference_started: Instant,
        inference_completed: Instant,
    ) -> Result<DetectionBatchV1> {
        ensure!(
            self.next_batch_sequence <= MAX_SAFE_JSON_INTEGER,
            "captured batch sequence exceeded the safe JSON integer range"
        );
        let detections = inference
            .detections
            .iter()
            .enumerate()
            .map(|(index, detection)| {
                let [left, top, right, bottom] = detection.bbox_xyxy_px;
                let bbox_image_px = BoundingBox {
                    left: left as f64,
                    top: top as f64,
                    right: right as f64,
                    bottom: bottom as f64,
                };
                Ok(crate::detection_batch::Detection {
                    detection_index: u64::try_from(index)
                        .context("captured detection index exceeds u64")?,
                    class_id: u32::try_from(detection.class_id)
                        .context("captured class ID exceeds u32")?,
                    class_name: detection.class_name.clone(),
                    confidence: detection.confidence as f64,
                    // Derive center from the serialized f64 box so the
                    // cross-runtime contract sees exact matching geometry.
                    center_image_px: Point {
                        x: (bbox_image_px.left + bbox_image_px.right) / 2.0,
                        y: (bbox_image_px.top + bbox_image_px.bottom) / 2.0,
                    },
                    bbox_image_px,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let source = frame.source();
        let batch = DetectionBatchV1 {
            schema_version: 1,
            engine_instance_id: self.config.engine_instance_id.clone(),
            batch_sequence: self.next_batch_sequence,
            source: BatchSourceFrame {
                session_id: source.session_id.clone(),
                frame_id: source.frame_id.clone(),
                timestamp: Some(source_timestamp),
                width_px: source.width_px,
                height_px: source.height_px,
            },
            image_space: self.config.image_space,
            model: self.detector.model_identity().clone(),
            settings: self.detector.inference_settings().clone(),
            inference_started_at: batch_monotonic_timestamp(
                self.clock_origin,
                inference_started,
                &self.config.engine_instance_id,
                BatchTimestampStage::InferenceStarted,
            )?,
            inference_completed_at: batch_monotonic_timestamp(
                self.clock_origin,
                inference_completed,
                &self.config.engine_instance_id,
                BatchTimestampStage::InferenceComplete,
            )?,
            published_at: batch_monotonic_timestamp(
                self.clock_origin,
                Instant::now(),
                &self.config.engine_instance_id,
                BatchTimestampStage::BatchPublication,
            )?,
            outcome: BatchOutcome::Success,
            reason: None,
            detections,
        };
        batch
            .validate()
            .context("captured detection batch violates the v1 contract")?;
        Ok(batch)
    }
}

fn monotonic_elapsed_ns(origin: Instant, at: Instant) -> Result<u128> {
    at.checked_duration_since(origin)
        .map(|duration| duration.as_nanos())
        .context("monotonic timestamp predates the capture pipeline clock origin")
}

fn validate_config(config: &CapturePipelineConfig, tracking: &TrackingConfigV1) -> Result<()> {
    ensure!(
        !config.engine_instance_id.is_empty()
            && config.engine_instance_id.chars().count()
                <= crate::detection_batch::MAX_IDENTIFIER_CHARACTERS
            && !config.engine_instance_id.chars().any(char::is_control),
        "engine instance ID must be a bounded non-empty identifier"
    );
    ensure!(
        !config.maximum_frame_age.is_zero(),
        "maximum captured-frame age must be positive"
    );
    ensure!(
        config.capture_configuration_revision <= MAX_SAFE_JSON_INTEGER,
        "capture configuration revision exceeds the exact-integer range of common JSON clients"
    );
    if let Some(calibration) = &config.calibration {
        ensure!(
            calibration.camera_profile_id == tracking.camera_profile_id(),
            "applied calibration camera profile does not match tracking configuration"
        );
        ensure!(
            calibration.rectification_revision == tracking.rectification_revision(),
            "applied calibration rectification revision does not match tracking configuration"
        );
        let configured_image_space = match config.image_space {
            BatchImageSpace::Raw => crate::ImageSpace::Raw,
            BatchImageSpace::Rectified => crate::ImageSpace::Rectified,
        };
        ensure!(
            calibration.calibration.image_space() == configured_image_space,
            "applied calibration image space does not match captured frame configuration"
        );
    }
    Ok(())
}

fn frame_age(read_completed_at: Instant, now: Instant) -> Result<Duration> {
    now.checked_duration_since(read_completed_at)
        .context("host frame read-completion time is later than the local monotonic clock")
}

fn start_timing_child_span(
    context: Option<&FrameTimingContext>,
    name: &str,
    parent: Option<SpanId>,
) -> Option<SpanGuard> {
    let (Some(context), Some(parent)) = (context, parent) else {
        return None;
    };
    context.start_child_span(name, parent).ok()
}

fn batch_monotonic_timestamp(
    origin: Instant,
    at: Instant,
    domain_id: &str,
    stage: BatchTimestampStage,
) -> Result<BatchTimestamp> {
    let elapsed = at
        .checked_duration_since(origin)
        .context("capture pipeline clock origin must precede every captured event")?;
    let ticks = u64::try_from(elapsed.as_nanos())
        .context("process-local nanosecond timestamp exceeded the v1 range")?;
    Ok(BatchTimestamp {
        ticks: ticks.to_string(),
        unit: BatchTimeUnit::Nanoseconds,
        clock: BatchClockKind::Monotonic,
        domain_id: domain_id.to_owned(),
        stage,
    })
}

fn scene_publication_timestamp(
    origin: Instant,
    domain_id: &str,
    at: Instant,
) -> Result<SceneTimestamp> {
    let elapsed = at
        .checked_duration_since(origin)
        .context("capture pipeline clock origin must precede scene publication")?;
    let ticks = u64::try_from(elapsed.as_nanos())
        .context("process-local scene timestamp exceeded the v1 range")?;
    Ok(SceneTimestamp {
        ticks: ticks.to_string(),
        unit: TimeUnit::Nanoseconds,
        clock: ClockKind::Monotonic,
        domain_id: domain_id.to_owned(),
        stage: SceneTimestampStage::SnapshotPublication,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::calibration::{TableCalibrationV1, ValidatedTableCalibrationV1};
    use crate::capture::{
        CaptureAggregateFrameLimit, CaptureFrameBudget, CaptureFrameLimit, RgbFrameEnvelope,
    };
    use crate::detection_batch::{
        BatchOutcome, BatchTimestamp, ClockKind as BatchClockKind, ImageSpace as BatchImageSpace,
        InferenceSettings, ModelClass, ModelIdentity, SourceFrame, TimeUnit as BatchTimeUnit,
        TimestampStage as BatchTimestampStage,
    };
    use crate::inference::{Detection, FrameInferenceOutput, FrameInferenceTimings};
    use crate::live_audit::{
        LiveAuditConfig, LiveAuditExporter, LiveAuditRecorder, LiveAuditShutdownSummary,
    };
    use crate::scene_transport::SceneTransportConfig;
    use crate::tracking::TrackingConfigV1;
    use crate::{Outcome, UpdateKind};
    use serde_json::{json, Value};

    use super::{
        AppliedCaptureCalibration, CapturePipelineConfig, CapturePipelineOutcome,
        CaptureScenePipeline, CaptureSkipReason, Rgb8FrameDetector,
    };

    struct FakeDetector {
        model: ModelIdentity,
        settings: InferenceSettings,
        detections: Vec<Detection>,
        detection_batches: Option<Vec<Vec<Detection>>>,
        calls: Arc<AtomicU64>,
        fail: bool,
        delay: Duration,
    }

    impl Rgb8FrameDetector for FakeDetector {
        fn infer_rgb8(
            &mut self,
            width_px: u32,
            height_px: u32,
            _pixels: &[u8],
        ) -> anyhow::Result<FrameInferenceOutput> {
            let call_index = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            if self.fail {
                anyhow::bail!("injected detector failure");
            }
            Ok(FrameInferenceOutput {
                source_width_px: width_px,
                source_height_px: height_px,
                detections: self
                    .detection_batches
                    .as_ref()
                    .and_then(|batches| batches.get(call_index))
                    .unwrap_or(&self.detections)
                    .clone(),
                timings: FrameInferenceTimings {
                    preprocess_ms: 1.0,
                    inference_ms: 2.0,
                    decode_and_nms_ms: 0.5,
                    engine_total_ms: 3.5,
                },
            })
        }

        fn model_identity(&self) -> &ModelIdentity {
            &self.model
        }

        fn inference_settings(&self) -> &InferenceSettings {
            &self.settings
        }
    }

    fn detection() -> Detection {
        Detection {
            class_id: 1,
            class_name: "ball".to_owned(),
            confidence: 0.91,
            bbox_xyxy_px: [3.0, 4.0, 9.0, 10.0],
            center_xy_px: [6.0, 7.0],
        }
    }

    fn detection_at(x: f32, y: f32) -> Detection {
        Detection {
            center_xy_px: [x, y],
            bbox_xyxy_px: [x - 1.0, y - 1.0, x + 1.0, y + 1.0],
            ..detection()
        }
    }

    fn detector(calls: Arc<AtomicU64>, fail: bool, detections: Vec<Detection>) -> FakeDetector {
        FakeDetector {
            model: ModelIdentity {
                artifact_sha256: "a".repeat(64),
                source_checkpoint_sha256: Some("b".repeat(64)),
                classes: vec![
                    ModelClass {
                        class_id: 0,
                        class_name: "cue".to_owned(),
                    },
                    ModelClass {
                        class_id: 1,
                        class_name: "ball".to_owned(),
                    },
                ],
            },
            settings: InferenceSettings {
                input_width_px: 32,
                input_height_px: 32,
                confidence_threshold: 0.4,
                iou_threshold: 0.45,
                max_detections: 32,
            },
            detections,
            detection_batches: None,
            calls,
            fail,
            delay: Duration::ZERO,
        }
    }

    fn slow_detector(calls: Arc<AtomicU64>, delay: Duration) -> FakeDetector {
        let mut detector = detector(calls, false, vec![detection()]);
        detector.delay = delay;
        detector
    }

    fn fixture_calibration() -> ValidatedTableCalibrationV1 {
        serde_json::from_str::<TableCalibrationV1>(include_str!(
            "../contracts/calibration/v1/table-calibration-center-origin.json"
        ))
        .unwrap()
        .into_validated()
        .unwrap()
    }

    fn pipeline(
        detector: FakeDetector,
        maximum_frame_age: Duration,
        maximum_message_bytes: usize,
    ) -> CaptureScenePipeline<FakeDetector> {
        pipeline_with_delivery(
            detector,
            maximum_frame_age,
            maximum_message_bytes,
            false,
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
        )
    }

    fn pipeline_with_delivery(
        detector: FakeDetector,
        maximum_frame_age: Duration,
        maximum_message_bytes: usize,
        public_state_delivery_enabled: bool,
        address: SocketAddr,
    ) -> CaptureScenePipeline<FakeDetector> {
        let tracking = TrackingConfigV1::new(
            1,
            Some(0),
            "camera-main",
            40.0,
            Duration::from_millis(100),
            Duration::from_secs(1),
            Duration::from_secs(3),
            32,
            8,
            1,
            1,
        )
        .unwrap();
        CaptureScenePipeline::new(
            detector,
            tracking,
            CapturePipelineConfig {
                engine_instance_id: "capture-run-1".to_owned(),
                capture_configuration_revision: 1,
                image_space: BatchImageSpace::Raw,
                maximum_frame_age,
                calibration: None,
                scene_transport: SceneTransportConfig {
                    address,
                    maximum_message_bytes,
                    public_state_delivery_enabled,
                    connect_timeout: Duration::from_millis(20),
                    write_timeout: Duration::from_millis(20),
                    reconnect_initial_delay: Duration::from_millis(20),
                    reconnect_maximum_delay: Duration::from_millis(40),
                },
            },
        )
        .unwrap()
    }

    fn audited_pipeline(
        detector: FakeDetector,
        maximum_frame_age: Duration,
        audit: LiveAuditRecorder,
    ) -> CaptureScenePipeline<FakeDetector> {
        let tracking = TrackingConfigV1::new(
            1,
            Some(0),
            "camera-main",
            40.0,
            Duration::from_millis(100),
            Duration::from_secs(1),
            Duration::from_secs(3),
            32,
            8,
            1,
            1,
        )
        .unwrap();
        CaptureScenePipeline::new_with_audit(
            detector,
            tracking,
            CapturePipelineConfig {
                engine_instance_id: "capture-run-outcomes".to_owned(),
                capture_configuration_revision: 7,
                image_space: BatchImageSpace::Raw,
                maximum_frame_age,
                calibration: None,
                scene_transport: SceneTransportConfig {
                    address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
                    maximum_message_bytes: 16 * 1024,
                    public_state_delivery_enabled: false,
                    connect_timeout: Duration::from_millis(20),
                    write_timeout: Duration::from_millis(20),
                    reconnect_initial_delay: Duration::from_millis(20),
                    reconnect_maximum_delay: Duration::from_millis(40),
                },
            },
            Some(audit),
        )
        .unwrap()
    }

    fn frame(frame_id: &str, ticks: Option<u64>, read_at: Instant) -> RgbFrameEnvelope {
        let width_px = 16;
        let height_px = 16;
        let frame_limit = CaptureFrameLimit::new((width_px * height_px * 3) as usize).unwrap();
        let budget = CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(2_048).unwrap());
        let pixels = budget
            .allocate_rgb8_buffer(width_px, height_px, frame_limit)
            .unwrap();
        let timestamp = ticks.map(|ticks| BatchTimestamp {
            ticks: ticks.to_string(),
            unit: BatchTimeUnit::Nanoseconds,
            clock: BatchClockKind::Sensor,
            domain_id: "camera-clock-1".to_owned(),
            stage: BatchTimestampStage::SharedMemoryPublication,
        });
        RgbFrameEnvelope::new(
            SourceFrame {
                session_id: "source-session-1".to_owned(),
                frame_id: frame_id.to_owned(),
                timestamp,
                width_px,
                height_px,
            },
            pixels,
            read_at,
            frame_limit,
        )
        .unwrap()
    }

    fn read_transport_frame(stream: &mut impl Read) -> Vec<u8> {
        let mut header = [0_u8; 4];
        stream.read_exact(&mut header).unwrap();
        let mut payload = vec![0; u32::from_be_bytes(header) as usize];
        stream.read_exact(&mut payload).unwrap();
        payload
    }

    #[test]
    fn live_public_state_delivery_contains_rust_lifecycle_and_unchanged_raw_scene() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let hello: Value = serde_json::from_slice(&read_transport_frame(&mut stream)).unwrap();
            assert_eq!(hello["protocol_version"], 2);
            let ready = serde_json::to_vec(&json!({
                "type": "ready",
                "protocol_version": 2,
                "last_transport_sequence": null
            }))
            .unwrap();
            crate::framing::write_framed_message(&mut stream, &ready, 64 * 1024).unwrap();
            serde_json::from_slice::<Value>(&read_transport_frame(&mut stream)).unwrap()
        });

        let calls = Arc::new(AtomicU64::new(0));
        let mut pipeline = pipeline_with_delivery(
            detector(Arc::clone(&calls), false, vec![detection()]),
            Duration::from_secs(1),
            64 * 1024,
            true,
            address,
        );
        let outcome = pipeline
            .process(&frame(
                "native-state-frame",
                Some(1_000_000),
                Instant::now(),
            ))
            .unwrap();
        assert!(matches!(outcome, CapturePipelineOutcome::Published(_)));
        let event = receiver.join().unwrap();
        assert_eq!(event["delivery_type"], "billiards_public_state_delivery");
        assert_eq!(event["delivery_schema_version"], 1);
        assert_eq!(event["source_snapshot"]["schema_version"], 1);
        assert_eq!(
            event["source_snapshot"]["source"]["frame_id"],
            "native-state-frame"
        );
        assert_eq!(event["current_state"]["freshness"], "fresh");
        assert_eq!(event["current_state"]["epoch"], 1);
        assert_eq!(
            event["current_state"]["payload"]["balls"][0]["id"],
            "track-1"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        pipeline.shutdown();
    }

    #[test]
    fn captured_frame_is_correlated_into_a_complete_scene_and_queued() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut pipeline = pipeline(
            detector(Arc::clone(&calls), false, vec![detection()]),
            Duration::from_secs(1),
            16 * 1024,
        );
        let source_time = BatchTimestamp {
            ticks: "900000".to_owned(),
            unit: BatchTimeUnit::Nanoseconds,
            clock: BatchClockKind::Sensor,
            domain_id: "camera-clock-1".to_owned(),
            stage: BatchTimestampStage::SharedMemoryPublication,
        };
        let captured = frame("frame-alpha", Some(900_000), Instant::now());
        let outcome = pipeline.process(&captured).unwrap();
        let CapturePipelineOutcome::Published(result) = outcome else {
            panic!("expected a captured scene");
        };

        assert_eq!(result.batch.source.session_id, "source-session-1");
        assert_eq!(result.batch.source.frame_id, "frame-alpha");
        assert_eq!(result.batch.source.timestamp, Some(source_time));
        assert_eq!(result.batch.detections[0].center_image_px.x, 6.0);
        assert_eq!(result.batch.detections[0].center_image_px.y, 7.0);
        assert_eq!(result.scene.source.session_id, "source-session-1");
        assert_eq!(result.scene.source.frame_id, "frame-alpha");
        assert_eq!(
            result.scene.source.timestamp.stage,
            crate::TimestampStage::SharedMemoryPublication
        );
        assert_eq!(result.scene.update_kind, UpdateKind::Observation);
        assert_eq!(result.scene.outcome, Outcome::Success);
        assert_eq!(result.scene.balls.len(), 1);
        assert_eq!(result.calibration_ms, None);
        assert!(result.scene.balls[0].identity.number.is_none());
        assert!(result
            .parent_record_read_complete_to_inference_start_ms
            .is_finite());
        assert!(
            result.parent_record_read_complete_to_scene_queue_ms
                >= result.parent_record_read_complete_to_inference_start_ms
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(pipeline.transport_stats().queued_snapshots, 1);
        assert_eq!(captured.rgb8().len(), 16 * 16 * 3);
        pipeline.shutdown();
    }

    #[test]
    fn association_capacity_pressure_publishes_current_locations_and_keeps_running() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut detector = detector(Arc::clone(&calls), false, Vec::new());
        detector.settings.max_detections = 2;
        detector.detection_batches = Some(vec![
            vec![detection_at(2.0, 8.0), detection_at(14.0, 8.0)],
            vec![detection_at(14.5, 8.0), detection_at(5.0, 8.0)],
            vec![detection_at(14.8, 8.0), detection_at(5.5, 8.0)],
        ]);
        let tracking = TrackingConfigV1::new(
            1,
            Some(0),
            "camera-main",
            2.5,
            Duration::from_millis(50),
            Duration::from_millis(100),
            Duration::from_millis(500),
            2,
            8,
            1,
            1,
        )
        .unwrap();
        let mut pipeline = CaptureScenePipeline::new(
            detector,
            tracking,
            CapturePipelineConfig {
                engine_instance_id: "capture-run-capacity-pressure".to_owned(),
                capture_configuration_revision: 1,
                image_space: BatchImageSpace::Raw,
                maximum_frame_age: Duration::from_secs(1),
                calibration: None,
                scene_transport: SceneTransportConfig {
                    address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
                    maximum_message_bytes: 16 * 1024,
                    public_state_delivery_enabled: false,
                    connect_timeout: Duration::from_millis(20),
                    write_timeout: Duration::from_millis(20),
                    reconnect_initial_delay: Duration::from_millis(20),
                    reconnect_maximum_delay: Duration::from_millis(40),
                },
            },
        )
        .unwrap();

        let first = pipeline
            .process(&frame(
                "capacity-frame-1",
                Some(1_000_000_000),
                Instant::now(),
            ))
            .unwrap();
        let CapturePipelineOutcome::Published(first) = first else {
            panic!("expected first complete scene");
        };
        assert_eq!(first.scene.balls.len(), 2);
        assert_eq!(first.scene.outcome, Outcome::Success);

        let pressure = pipeline
            .process(&frame(
                "capacity-frame-2",
                Some(1_010_000_000),
                Instant::now(),
            ))
            .unwrap();
        let CapturePipelineOutcome::Published(pressure) = pressure else {
            panic!("capacity pressure must retain the current scene");
        };
        assert_eq!(pressure.scene.balls.len(), 2);
        assert_eq!(pressure.association_capacity_evictions, 1);
        assert_eq!(pressure.scene.outcome, Outcome::Degraded);
        assert_eq!(
            pressure.scene.reason.as_deref(),
            Some("association_capacity_pressure")
        );
        assert_eq!(pressure.scene.balls[0].track_id, "track-2");
        assert_eq!(pressure.scene.balls[1].track_id, "track-3");
        assert!(pressure
            .scene
            .balls
            .iter()
            .all(|ball| ball.status == crate::BallStatus::Observed));

        let recovered = pipeline
            .process(&frame(
                "capacity-frame-3",
                Some(1_020_000_000),
                Instant::now(),
            ))
            .unwrap();
        let CapturePipelineOutcome::Published(recovered) = recovered else {
            panic!("the source must keep publishing after association eviction");
        };
        assert_eq!(recovered.scene.outcome, Outcome::Success);
        assert_eq!(
            recovered
                .scene
                .balls
                .iter()
                .map(|ball| ball.track_id.as_str())
                .collect::<Vec<_>>(),
            ["track-2", "track-3"]
        );
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        let stats = pipeline.stats();
        assert_eq!(stats.scenes_submitted, 3);
        assert_eq!(stats.association_capacity_pressure_frames, 1);
        assert_eq!(stats.association_capacity_evictions, 1);
        pipeline.shutdown();
    }

    #[test]
    fn pipeline_rejects_track_capacity_below_detector_output_bound_before_capture() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut detector = detector(Arc::clone(&calls), false, Vec::new());
        detector.settings.max_detections = 2;
        let tracking = TrackingConfigV1::new(
            1,
            Some(0),
            "camera-main",
            2.5,
            Duration::from_millis(50),
            Duration::from_millis(100),
            Duration::from_millis(500),
            1,
            8,
            1,
            1,
        )
        .unwrap();

        let pipeline = CaptureScenePipeline::new(
            detector,
            tracking,
            CapturePipelineConfig {
                engine_instance_id: "capture-run-invalid-capacity".to_owned(),
                capture_configuration_revision: 1,
                image_space: BatchImageSpace::Raw,
                maximum_frame_age: Duration::from_secs(1),
                calibration: None,
                scene_transport: SceneTransportConfig {
                    address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
                    maximum_message_bytes: 16 * 1024,
                    public_state_delivery_enabled: false,
                    connect_timeout: Duration::from_millis(20),
                    write_timeout: Duration::from_millis(20),
                    reconnect_initial_delay: Duration::from_millis(20),
                    reconnect_maximum_delay: Duration::from_millis(40),
                },
            },
        );

        assert!(pipeline.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn opted_in_pipeline_exports_measured_stage_spans_and_receiver_queue_wait() {
        let listener =
            std::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
                .unwrap();
        let audit_address = listener.local_addr().unwrap();
        let audit_receiver = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let frame_record = read_audit_record(&mut stream);
            let footer_record = read_audit_record(&mut stream);
            (frame_record, footer_record)
        });
        let audit = LiveAuditExporter::start(LiveAuditConfig {
            address: audit_address,
            queue_capacity: 4,
            maximum_record_bytes: 16 * 1024,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
        })
        .unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let tracking = TrackingConfigV1::new(
            1,
            Some(0),
            "camera-main",
            40.0,
            Duration::from_millis(100),
            Duration::from_secs(1),
            Duration::from_secs(3),
            32,
            8,
            1,
            1,
        )
        .unwrap();
        let mut pipeline = CaptureScenePipeline::new_with_audit(
            detector(Arc::clone(&calls), false, vec![detection()]),
            tracking,
            CapturePipelineConfig {
                engine_instance_id: "capture-run-audit".to_owned(),
                capture_configuration_revision: 1,
                image_space: BatchImageSpace::Raw,
                maximum_frame_age: Duration::from_secs(1),
                calibration: None,
                scene_transport: SceneTransportConfig {
                    address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
                    maximum_message_bytes: 16 * 1024,
                    public_state_delivery_enabled: false,
                    connect_timeout: Duration::from_millis(20),
                    write_timeout: Duration::from_millis(20),
                    reconnect_initial_delay: Duration::from_millis(20),
                    reconnect_maximum_delay: Duration::from_millis(40),
                },
            },
            Some(audit.recorder()),
        )
        .unwrap();

        let captured = frame("audit-frame", Some(900_000), Instant::now());
        let outcome = pipeline
            .process_with_receiver_queue_wait(&captured, Some(5_000_000))
            .unwrap();
        assert!(matches!(outcome, CapturePipelineOutcome::Published(_)));
        let pipeline_counters = pipeline.stats();
        let scene_transport_counters = pipeline.shutdown();
        drop(pipeline);
        let summary = audit.finish_with_shutdown_summary(LiveAuditShutdownSummary {
            clean_shutdown: true,
            pipeline_counters: Some(pipeline_counters),
            scene_transport_counters: Some(scene_transport_counters),
        });
        let (frame_record, footer_record) = audit_receiver.join().unwrap();

        assert_eq!(summary.failure, None);
        assert_eq!(frame_record["record_type"], "frame");
        assert_eq!(frame_record["engine_instance_id"], "capture-run-audit");
        assert_eq!(frame_record["source_session_id"], "source-session-1");
        assert_eq!(frame_record["frame_id"], "audit-frame");
        assert_eq!(frame_record["transport_sequence"], 1);
        assert_eq!(frame_record["configuration_revision"], 1);
        assert_eq!(frame_record["tracking_configuration_revision"], 1);
        assert_eq!(frame_record["rectification_revision"], 1);
        assert_eq!(
            frame_record["calibration_revision"],
            serde_json::Value::Null
        );
        assert_eq!(frame_record["detection_count"], 1);
        assert_eq!(frame_record["track_count"], 1);
        assert_eq!(
            frame_record["source_timestamp"]["stage"],
            "shared_memory_publication"
        );
        let durations = &frame_record["durations_ms"];
        assert_eq!(durations["preprocess"], 1.0);
        assert_eq!(durations["inference"], 2.0);
        assert_eq!(durations["decode_and_nms"], 0.5);
        assert_eq!(durations["capture_receiver_queue_wait"], 5.0);
        assert_eq!(durations["calibration"], serde_json::Value::Null);
        for stage in [
            "tracking_and_state_assembly",
            "scene_json_serialization",
            "scene_queue_submit",
        ] {
            assert!(durations[stage].as_f64().is_some_and(|value| value >= 0.0));
        }
        let timing = &frame_record["frame_timing"];
        assert!(timing["frame_elapsed_ns"].as_u64().is_some());
        let spans = timing["spans"].as_array().unwrap();
        assert!(spans.iter().any(|span| span["name"] == "frame_processing"));
        assert!(spans
            .iter()
            .any(|span| span["name"] == "capture_receiver_queue_wait"));
        assert!(spans.iter().any(|span| span["name"] == "detector"));
        assert!(spans
            .iter()
            .any(|span| span["name"] == "tracking_and_state_assembly"));
        assert!(spans
            .iter()
            .all(|span| span["inclusive_elapsed_ns"].as_u64().is_some()));
        assert!(frame_record.get("balls").is_none());
        assert!(frame_record.get("detections").is_none());
        assert_eq!(footer_record["record_type"], "footer");
        assert_eq!(footer_record["export_status"], "complete");
        assert_eq!(footer_record["clean_shutdown"], true);
        assert_eq!(footer_record["pipeline_counters"]["scenes_submitted"], 1);
        assert!(footer_record.get("scene_transport_counters").is_some());
    }

    #[test]
    fn opted_in_pipeline_exports_correlated_skipped_and_failed_observations() {
        let listener =
            std::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
                .unwrap();
        let audit_address = listener.local_addr().unwrap();
        let audit_receiver = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            (0..7)
                .map(|_| read_audit_record(&mut stream))
                .collect::<Vec<_>>()
        });
        let audit = LiveAuditExporter::start(LiveAuditConfig {
            address: audit_address,
            queue_capacity: 8,
            maximum_record_bytes: 16 * 1024,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
        })
        .unwrap();
        let recorder = audit.recorder();
        let calls = Arc::new(AtomicU64::new(0));
        let mut inference_failure = audited_pipeline(
            detector(Arc::clone(&calls), true, vec![]),
            Duration::from_secs(1),
            recorder.clone(),
        );
        let failed = frame("inference-failure", Some(400), Instant::now());
        assert!(inference_failure.process(&failed).is_err());
        assert!(matches!(
            inference_failure.process(&failed).unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::DuplicateSourceFrame)
        ));
        let nonincreasing = frame("non-increasing", Some(399), Instant::now());
        assert!(matches!(
            inference_failure.process(&nonincreasing).unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::NonIncreasingSourceTime)
        ));
        assert_eq!(inference_failure.stats().inference_failures, 1);
        inference_failure.shutdown();
        drop(inference_failure);

        let mut stale_pipeline = audited_pipeline(
            detector(Arc::clone(&calls), false, vec![detection()]),
            Duration::from_millis(1),
            recorder.clone(),
        );
        let stale = frame(
            "stale-before-inference",
            Some(401),
            Instant::now() - Duration::from_millis(50),
        );
        assert!(matches!(
            stale_pipeline.process(&stale).unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::StaleBeforeInference)
        ));
        stale_pipeline.shutdown();
        drop(stale_pipeline);

        let mut stale_after_inference = audited_pipeline(
            slow_detector(Arc::clone(&calls), Duration::from_millis(20)),
            Duration::from_millis(1),
            recorder.clone(),
        );
        assert!(matches!(
            stale_after_inference
                .process(&frame("stale-after-inference", Some(402), Instant::now()))
                .unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::StaleAfterInference)
        ));
        stale_after_inference.shutdown();
        drop(stale_after_inference);

        let mut invalid_detection = detection();
        invalid_detection.bbox_xyxy_px[2] = f32::NAN;
        let mut scene_failure = audited_pipeline(
            detector(Arc::clone(&calls), false, vec![invalid_detection]),
            Duration::from_secs(1),
            recorder,
        );
        assert!(scene_failure
            .process(&frame("scene-failure", Some(403), Instant::now()))
            .is_err());
        let pipeline_counters = scene_failure.stats();
        let scene_transport_counters = scene_failure.shutdown();
        drop(scene_failure);

        let summary = audit.finish_with_shutdown_summary(LiveAuditShutdownSummary {
            clean_shutdown: false,
            pipeline_counters: Some(pipeline_counters),
            scene_transport_counters: Some(scene_transport_counters),
        });
        let records = audit_receiver.join().unwrap();

        assert_eq!(summary.failure, None);
        assert_eq!(summary.skipped_observations, 4);
        assert_eq!(summary.failed_observations, 2);
        assert_eq!(records[0]["record_type"], "observation_outcome");
        assert_eq!(records[0]["engine_instance_id"], "capture-run-outcomes");
        assert_eq!(records[0]["source_session_id"], "source-session-1");
        assert_eq!(records[0]["frame_id"], "inference-failure");
        assert_eq!(records[0]["transport_sequence"], serde_json::Value::Null);
        assert_eq!(records[0]["outcome"], "failed");
        assert_eq!(records[0]["stage"], "inference");
        assert_eq!(records[0]["reason"], "inference_failed");
        assert_eq!(
            records[0]["source_timestamp"]["stage"],
            "shared_memory_publication"
        );
        assert_eq!(records[1]["frame_id"], "inference-failure");
        assert_eq!(records[1]["outcome"], "skipped");
        assert_eq!(records[1]["reason"], "duplicate_source_frame");
        assert_eq!(records[2]["frame_id"], "non-increasing");
        assert_eq!(records[2]["reason"], "non_increasing_source_time");
        assert_eq!(records[3]["frame_id"], "stale-before-inference");
        assert_eq!(records[3]["reason"], "stale_before_inference");
        assert_eq!(records[4]["frame_id"], "stale-after-inference");
        assert_eq!(records[4]["reason"], "stale_after_inference");
        assert_eq!(
            records[4]["schema_version"],
            crate::live_audit::LIVE_AUDIT_SCHEMA_VERSION
        );
        assert_eq!(records[4]["transport_sequence"], serde_json::Value::Null);
        assert_eq!(
            records[4]["source_timestamp"]["stage"],
            "shared_memory_publication"
        );
        assert_eq!(records[4]["inference_timings_ms"]["preprocess"], 1.0);
        assert_eq!(records[4]["inference_timings_ms"]["inference"], 2.0);
        assert_eq!(records[4]["inference_timings_ms"]["decode_and_nms"], 0.5);
        assert_eq!(records[4]["inference_timings_ms"]["engine_total"], 3.5);
        assert!(records[4]["inference_timings_ms"]
            ["parent_record_read_complete_to_inference_complete"]
            .as_f64()
            .is_some_and(|age_ms| age_ms >= 20.0));
        assert_eq!(records[4]["inference_timings_ms"]["maximum_frame_age"], 1.0);
        assert!(records[4]["inference_timings_ms"]["parent_framed_record_read"].is_null());
        assert!(records[4].get("detections").is_none());
        assert_eq!(records[5]["frame_id"], "scene-failure");
        assert_eq!(records[5]["outcome"], "failed");
        assert_eq!(records[5]["stage"], "batch_assembly");
        assert_eq!(records[5]["reason"], "batch_assembly_failed");
        assert_eq!(records[6]["record_type"], "footer");
        assert_eq!(records[6]["skipped_observations"], 4);
        assert_eq!(records[6]["failed_observations"], 2);
        assert_eq!(
            records[6]["pipeline_counters"]["scene_processing_failures"],
            1
        );
    }

    fn read_audit_record(stream: &mut std::net::TcpStream) -> serde_json::Value {
        let mut header = [0_u8; 4];
        stream.read_exact(&mut header).unwrap();
        let mut payload = vec![0; u32::from_be_bytes(header) as usize];
        stream.read_exact(&mut payload).unwrap();
        serde_json::from_slice(&payload).unwrap()
    }

    #[test]
    fn applied_calibration_must_match_bundle_and_live_tracking_identity() {
        let fixture = fixture_calibration();
        assert!(
            AppliedCaptureCalibration::new(fixture.clone(), 4, "test-camera-profile", 2,).is_err()
        );
        assert!(
            AppliedCaptureCalibration::new(fixture.clone(), 3, "different-camera-profile", 2,)
                .is_err()
        );

        let applied = AppliedCaptureCalibration::new(fixture, 3, "test-camera-profile", 2).unwrap();
        let calls = Arc::new(AtomicU64::new(0));
        let tracking = TrackingConfigV1::new(
            1,
            Some(0),
            "camera-main",
            40.0,
            Duration::from_millis(100),
            Duration::from_secs(1),
            Duration::from_secs(3),
            32,
            8,
            1,
            1,
        )
        .unwrap();
        let invalid_config = CapturePipelineConfig {
            engine_instance_id: "capture-run-invalid-calibration".to_owned(),
            capture_configuration_revision: 1,
            image_space: BatchImageSpace::Rectified,
            maximum_frame_age: Duration::from_secs(1),
            calibration: Some(applied),
            scene_transport: SceneTransportConfig {
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
                maximum_message_bytes: 16 * 1024,
                public_state_delivery_enabled: false,
                connect_timeout: Duration::from_millis(20),
                write_timeout: Duration::from_millis(20),
                reconnect_initial_delay: Duration::from_millis(20),
                reconnect_maximum_delay: Duration::from_millis(40),
            },
        };

        assert!(CaptureScenePipeline::new(
            detector(Arc::clone(&calls), false, vec![detection()]),
            tracking,
            invalid_config,
        )
        .is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn missing_source_time_uses_labeled_host_read_completion_fallback() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut pipeline = pipeline(
            detector(Arc::clone(&calls), false, vec![detection()]),
            Duration::from_secs(1),
            16 * 1024,
        );
        let captured = frame("frame-host-time", None, Instant::now());
        let CapturePipelineOutcome::Published(result) = pipeline.process(&captured).unwrap() else {
            panic!("expected a captured scene");
        };

        let fallback = result.batch.source.timestamp.unwrap();
        assert_eq!(fallback.clock, BatchClockKind::Monotonic);
        assert_eq!(fallback.domain_id, "capture-run-1");
        assert_eq!(fallback.stage, BatchTimestampStage::CameraReadComplete);
        assert_ne!(fallback.stage, BatchTimestampStage::Exposure);
        assert_eq!(
            result.scene.source.timestamp.stage,
            crate::TimestampStage::CameraReadComplete
        );
        pipeline.shutdown();
    }

    #[test]
    fn duplicate_and_nonincreasing_frames_are_rejected_before_inference() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut pipeline = pipeline(
            detector(Arc::clone(&calls), false, vec![detection()]),
            Duration::from_secs(1),
            16 * 1024,
        );
        let first = frame("frame-first", Some(100), Instant::now());
        assert!(matches!(
            pipeline.process(&first).unwrap(),
            CapturePipelineOutcome::Published(_)
        ));
        assert!(matches!(
            pipeline.process(&first).unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::DuplicateSourceFrame)
        ));
        let older = frame("frame-older", Some(99), Instant::now());
        assert!(matches!(
            pipeline.process(&older).unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::NonIncreasingSourceTime)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(pipeline.stats().duplicate_frames, 1);
        assert_eq!(pipeline.stats().out_of_order_frames, 1);
        pipeline.shutdown();
    }

    #[test]
    fn stale_frames_and_inference_failures_do_not_publish_scenes() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut stale_pipeline = pipeline(
            detector(Arc::clone(&calls), false, vec![detection()]),
            Duration::from_millis(1),
            16 * 1024,
        );
        let stale = frame(
            "frame-stale",
            Some(100),
            Instant::now() - Duration::from_millis(50),
        );
        assert!(matches!(
            stale_pipeline.process(&stale).unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::StaleBeforeInference)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(stale_pipeline.transport_stats().queued_snapshots, 0);
        stale_pipeline.shutdown();

        let failed_calls = Arc::new(AtomicU64::new(0));
        let mut failed_pipeline = pipeline(
            detector(Arc::clone(&failed_calls), true, vec![]),
            Duration::from_secs(1),
            16 * 1024,
        );
        let failed = frame("frame-failed", Some(200), Instant::now());
        assert!(failed_pipeline.process(&failed).is_err());
        assert_eq!(failed_pipeline.stats().inference_failures, 1);
        assert_eq!(failed_pipeline.transport_stats().queued_snapshots, 0);
        failed_pipeline.shutdown();
    }

    #[test]
    fn stale_inference_result_is_rejected_after_the_compute_deadline() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut pipeline = pipeline(
            slow_detector(Arc::clone(&calls), Duration::from_millis(20)),
            Duration::from_millis(5),
            16 * 1024,
        );
        let captured = frame("frame-slow", Some(250), Instant::now());

        assert!(matches!(
            pipeline.process(&captured).unwrap(),
            CapturePipelineOutcome::Skipped(CaptureSkipReason::StaleAfterInference)
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(pipeline.stats().stale_frames, 1);
        assert_eq!(pipeline.transport_stats().queued_snapshots, 0);
        pipeline.shutdown();
    }

    #[test]
    fn successful_empty_detection_is_published_as_an_empty_observation() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut pipeline = pipeline(
            detector(Arc::clone(&calls), false, vec![]),
            Duration::from_secs(1),
            16 * 1024,
        );
        let empty = frame("frame-empty", Some(300), Instant::now());
        let CapturePipelineOutcome::Published(result) = pipeline.process(&empty).unwrap() else {
            panic!("successful empty detection should publish");
        };

        assert!(result.batch.detections.is_empty());
        assert!(result.scene.balls.is_empty());
        assert_eq!(result.batch.outcome, BatchOutcome::Success);
        assert_eq!(result.scene.outcome, Outcome::Success);
        assert_eq!(pipeline.transport_stats().queued_snapshots, 1);
        pipeline.shutdown();
    }

    #[test]
    fn oversized_scene_submission_is_rejected_before_pipeline_sequence_advances() {
        let calls = Arc::new(AtomicU64::new(0));
        // The message bound is intentionally below a full scene snapshot but
        // large enough for the protocol handshake.
        let mut pipeline = pipeline(
            detector(Arc::clone(&calls), false, vec![detection()]),
            Duration::from_secs(1),
            512,
        );
        let first = frame("frame-over-limit", Some(400), Instant::now());
        assert!(pipeline.process(&first).is_err());
        assert_eq!(pipeline.stats().publication_failures, 1);
        assert_eq!(pipeline.transport_stats().queued_snapshots, 0);
        assert_eq!(pipeline.next_batch_sequence, 1);
        pipeline.shutdown();
    }
}
