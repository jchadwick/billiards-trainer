use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use image::RgbImage;
use openvino::{Core, DeviceType, ElementType, InferRequest, PropertyKey, Tensor};
use ort::environment::Environment;
use ort::ep::{ExecutionProvider, CPU};
use ort::session::{
    builder::{AutoDevicePolicy, SessionBuilder},
    Session,
};
use ort::value::{TensorElementType, TensorRef, ValueType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::ball_localization::{
    refine_ball_batch, resolve_ball_measurements_for_batch, BallProposal, BallRefinementBatch,
};
use crate::ball_localization_profile::{BallLocalizationSettings, LoadedBallLocalizationProfile};
use crate::calibration::CalibrationContext;
use crate::cue_geometry::{
    estimate_cue_geometry, orient_cue_geometry_from_playing_surface, CueGeometry,
    CueGeometryConfig, CueProposal, SignedCueDirection,
};
use crate::detection_batch::{
    encode_detection_batch_v1, BatchOutcome, BatchTimestamp, BoundingBox, ClockKind,
    Detection as BatchDetection, DetectionBatchV1, ImageSpace, InferenceSettings, ModelClass,
    ModelIdentity, Point, SourceFrame as BatchSourceFrame, TimeUnit, TimestampStage,
    MAX_IDENTIFIER_CHARACTERS,
};
use crate::geometry::{calibrate_detection_batch, AppliedCalibrationContext, GeometryError};
use crate::live_timing_spans::{FrameTimingContext, SpanId};
use crate::motion_monitor::{MotionClassification, MotionMonitor, MotionMonitorConfig};
use crate::phase_scheduler::{
    CueEvidence, ImagePoint, LayoutObservation, MotionEvidence, Phase, PhaseScheduler,
    SchedulerConfig,
};
use crate::tracking::{SceneTrackerV1, TrackingConfigV1};
use crate::{
    calibration::{TableCalibrationV1, ValidatedTableCalibrationV1},
    encode_scene_snapshot_v1, ImageSpace as CalibrationImageSpace, SceneSnapshotV1,
};

use crate::video_timeline::{parse_ffprobe_timeline, VideoTimeline, VideoTimelineLimits};

use super::{
    decode_yolo_v8_output_with_coordinate_scale, prepare_rgb8, prepare_rgb_image, Detection,
    LetterboxGeometry, PreparedInput,
};

const INPUT_NAME: &str = "images";
const OUTPUT_NAME: &str = "output0";
const EXPECTED_CLASS_COUNT: usize = 2;
const MODEL_EXPORT_MANIFEST_SCHEMA_VERSION: u32 = 1;
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
/// Default per-video input size bound for recorded-video audit runs.
pub const DEFAULT_VIDEO_MAXIMUM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Default cap for metadata emitted by a full-stream ffprobe frame scan.
pub const DEFAULT_VIDEO_MAXIMUM_PROBE_JSON_BYTES: usize =
    crate::video_timeline::DEFAULT_MAXIMUM_PROBE_JSON_BYTES;
/// Default frame-count bound for PTS storage and replay work.
pub const DEFAULT_VIDEO_MAXIMUM_FRAME_COUNT: usize = 250_000;
/// Default maximum number of source pixels in one decoded video frame.
pub const DEFAULT_VIDEO_MAXIMUM_PIXEL_COUNT: u64 = 64 * 1024 * 1024;
/// Default cap for the temporary raw RGB decode spool.
pub const DEFAULT_VIDEO_MAXIMUM_DECODED_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// Default timeout for each ffprobe/ffmpeg child process.
pub const DEFAULT_VIDEO_SUBPROCESS_TIMEOUT_SECONDS: u64 = 600;
/// Default cap on optional per-frame detail lines; details are disabled unless requested.
pub const DEFAULT_VIDEO_MAXIMUM_DETAIL_RECORDS: usize = 1_000;
const MAXIMUM_VIDEO_CHILD_STDERR_BYTES: usize = 16 * 1024;

/// Model and post-processing settings for reusable per-frame inference.
///
/// This type intentionally contains no file-replay, camera, tracker, or
/// transport settings. A source adapter owns frame identity and timing, then
/// passes one immutable RGB image to the inference engine.
#[derive(Debug, Clone)]
pub struct FrameInferenceOptions {
    /// ONNX model used by CPU, auto, and OpenVINO GPU backends.
    pub onnx_path: Option<PathBuf>,
    pub source_checkpoint_path: PathBuf,
    pub provider: ProviderSelection,
    pub image_size: u32,
    pub confidence: f32,
    pub iou: f32,
    pub max_detections: usize,
    /// Intra-op thread hint for ONNX Runtime providers. Native OpenVINO GPU
    /// inference does not apply CPU thread tuning.
    pub intra_op_threads: Option<usize>,
}

/// Per-frame inference measurements. These durations exclude camera capture,
/// encoded-image decoding, queue wait, tracking, and publication.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FrameInferenceTimings {
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub decode_and_nms_ms: f64,
    pub engine_total_ms: f64,
}

/// One complete inference result for an owned or immutably borrowed RGB frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameInferenceOutput {
    pub source_width_px: u32,
    pub source_height_px: u32,
    pub detections: Vec<Detection>,
    pub timings: FrameInferenceTimings,
    /// Present on the live wrapper only after native-resolution refinement.
    /// The DetectionBatchV1 boxes and their exact raw midpoint remain intact.
    pub ball_refinements: Option<BallRefinementBatch>,
}

struct PreparedFrameInput {
    values: Vec<f32>,
    geometry: LetterboxGeometry,
}

impl From<PreparedInput> for PreparedFrameInput {
    fn from(prepared: PreparedInput) -> Self {
        Self {
            values: prepared.values,
            geometry: prepared.geometry,
        }
    }
}

/// Reusable inference session for single-frame RGB inference.
///
/// The mutable session is deliberately not shared: a live runtime should keep
/// this value on its inference worker and hand it only one frame at a time.
/// The environment field is declared after the session so the session is
/// dropped before its ONNX Runtime environment.
pub struct FrameInferenceEngine {
    session: InferenceSession,
    _environment: Option<Environment>,
    expected_output_shape: Vec<i64>,
    class_names: Vec<String>,
    output_coordinate_scale: f32,
    image_size: usize,
    confidence: f32,
    iou: f32,
    max_detections: usize,
    model_identity: ModelIdentity,
    inference_settings: InferenceSettings,
    requested_provider: ProviderSelection,
}

impl FrameInferenceEngine {
    /// Load a hash-verified model session using an environment owned by the
    /// application. One environment may be shared by multiple sessions.
    pub fn new(environment: Environment, options: FrameInferenceOptions) -> Result<Self> {
        Self::new_with_environment(Some(environment), options)
    }

    /// Load native OpenVINO GPU inference without initializing ONNX Runtime.
    /// The ONNX Runtime CPU and automatic providers require an environment
    /// created by the application.
    pub fn new_without_onnx_runtime(options: FrameInferenceOptions) -> Result<Self> {
        Self::new_with_environment(None, options)
    }

    fn new_with_environment(
        environment: Option<Environment>,
        options: FrameInferenceOptions,
    ) -> Result<Self> {
        validate_frame_inference_options(&options)?;
        let model_contract = load_and_verify_manifest(
            options
                .onnx_path
                .as_deref()
                .context("onnx_path is required for the selected provider")?,
            &options.source_checkpoint_path,
        )?;
        let expected_output_shape = vec![
            1_i64,
            (4 + model_contract.class_names.len()) as i64,
            expected_anchor_count(options.image_size)? as i64,
        ];
        let session = match options.provider {
            ProviderSelection::Openvino => InferenceSession::Openvino(OpenVinoSession::new(
                options
                    .onnx_path
                    .as_deref()
                    .context("onnx_path is required for the native OpenVINO GPU provider")?,
            )?),
            ProviderSelection::Cpu | ProviderSelection::Auto => {
                let environment = environment.as_ref().context(
                    "ONNX Runtime environment is required for CPU or automatic inference",
                )?;
                create_session_for_model(
                    environment,
                    options
                        .onnx_path
                        .as_deref()
                        .context("onnx_path is required for the selected provider")?,
                    options.provider,
                    options.intra_op_threads,
                    None,
                )?
            }
        };
        let actual_output_shape =
            validate_session(&session, options.image_size, &model_contract.class_names)?;
        ensure!(
            actual_output_shape == expected_output_shape,
            "model output signature does not match the requested inference profile: {actual_output_shape:?}"
        );
        let model_identity = ModelIdentity {
            artifact_sha256: model_contract.artifact_sha256.clone(),
            source_checkpoint_sha256: Some(model_contract.source_checkpoint_sha256),
            classes: model_contract
                .class_names
                .iter()
                .enumerate()
                .map(|(class_id, class_name)| ModelClass {
                    class_id: class_id as u32,
                    class_name: class_name.clone(),
                })
                .collect(),
        };
        let inference_settings = InferenceSettings {
            input_width_px: options.image_size,
            input_height_px: options.image_size,
            confidence_threshold: options.confidence as f64,
            iou_threshold: options.iou as f64,
            max_detections: options.max_detections as u32,
        };

        Ok(Self {
            session,
            _environment: environment,
            expected_output_shape,
            class_names: model_contract.class_names,
            output_coordinate_scale: model_contract.output_coordinate_scale,
            image_size: options.image_size as usize,
            confidence: options.confidence,
            iou: options.iou,
            max_detections: options.max_detections,
            model_identity,
            inference_settings,
            requested_provider: options.provider,
        })
    }

    /// Build from a model contract already verified by the caller and report
    /// only session construction and signature validation time. Audit setup
    /// verifies provenance before this interval so it stays comparable with
    /// the provider-probe session metric.
    fn from_verified_model(
        environment: Environment,
        options: FrameInferenceOptions,
        model_contract: LoadedModelContract,
    ) -> Result<(Self, f64)> {
        validate_frame_inference_options(&options)?;
        let expected_output_shape = vec![
            1_i64,
            (4 + model_contract.class_names.len()) as i64,
            expected_anchor_count(options.image_size)? as i64,
        ];
        let session_start = Instant::now();
        let session = create_session_for_model(
            &environment,
            options
                .onnx_path
                .as_deref()
                .context("ONNX model path is required for recorded-video inference")?,
            options.provider,
            options.intra_op_threads,
            None,
        )?;
        let actual_output_shape =
            validate_session(&session, options.image_size, &model_contract.class_names)?;
        ensure!(
            actual_output_shape == expected_output_shape,
            "ONNX output signature does not match the requested inference profile: {actual_output_shape:?}"
        );
        let session_init_ms = elapsed_ms(session_start);

        let model_identity = ModelIdentity {
            artifact_sha256: model_contract.artifact_sha256.clone(),
            source_checkpoint_sha256: Some(model_contract.source_checkpoint_sha256),
            classes: model_contract
                .class_names
                .iter()
                .enumerate()
                .map(|(class_id, class_name)| ModelClass {
                    class_id: class_id as u32,
                    class_name: class_name.clone(),
                })
                .collect(),
        };
        let inference_settings = InferenceSettings {
            input_width_px: options.image_size,
            input_height_px: options.image_size,
            confidence_threshold: options.confidence as f64,
            iou_threshold: options.iou as f64,
            max_detections: options.max_detections as u32,
        };

        Ok((
            Self {
                session,
                _environment: Some(environment),
                expected_output_shape,
                class_names: model_contract.class_names,
                output_coordinate_scale: model_contract.output_coordinate_scale,
                image_size: options.image_size as usize,
                confidence: options.confidence,
                iou: options.iou,
                max_detections: options.max_detections,
                model_identity,
                inference_settings,
                requested_provider: options.provider,
            },
            session_init_ms,
        ))
    }

    /// Run provider-specific preprocessing, inference, and YOLO decoding on one RGB frame.
    pub fn infer_rgb(&mut self, image: &RgbImage) -> Result<FrameInferenceOutput> {
        self.infer_rgb8(image.width(), image.height(), image.as_raw())
    }

    /// Run inference directly from packed, borrowed RGB8 bytes. This avoids
    /// materializing an `RgbImage` between capture and model preprocessing.
    pub fn infer_rgb8(
        &mut self,
        source_width: u32,
        source_height: u32,
        pixels: &[u8],
    ) -> Result<FrameInferenceOutput> {
        let engine_start = Instant::now();
        let preprocess_start = Instant::now();
        let prepared = self
            .prepare_rgb8_for_provider(source_width, source_height, pixels)
            .context("failed to preprocess RGB frame")?;
        let preprocess_ms = elapsed_ms(preprocess_start);

        let mut output = self.infer_input_with_spans(&prepared.values, prepared.geometry, None)?;
        output.timings.preprocess_ms = preprocess_ms;
        output.timings.engine_total_ms = elapsed_ms(engine_start);
        Ok(output)
    }

    /// Run borrowed RGB8 inference while recording its nested detector stages
    /// into the caller's opt-in frame context. The ordinary inference entry
    /// point remains free of audit allocations and locking.
    pub(crate) fn infer_rgb8_with_spans(
        &mut self,
        source_width: u32,
        source_height: u32,
        pixels: &[u8],
        timing: &FrameTimingContext,
        parent: SpanId,
    ) -> Result<FrameInferenceOutput> {
        let engine_start = Instant::now();
        let preprocess_start = Instant::now();
        let preprocess_span = timing.start_child_span("preprocess", parent).ok();
        let prepared = self
            .prepare_rgb8_for_provider(source_width, source_height, pixels)
            .context("failed to preprocess RGB frame")?;
        drop(preprocess_span);
        let preprocess_ms = elapsed_ms(preprocess_start);

        let mut output = self.infer_input_with_spans(
            &prepared.values,
            prepared.geometry,
            Some((timing, parent)),
        )?;
        output.timings.preprocess_ms = preprocess_ms;
        output.timings.engine_total_ms = elapsed_ms(engine_start);
        Ok(output)
    }

    fn infer_prepared(&mut self, prepared: &PreparedInput) -> Result<FrameInferenceOutput> {
        self.infer_prepared_with_spans(prepared, None)
    }

    fn infer_prepared_with_spans(
        &mut self,
        prepared: &PreparedInput,
        timing: Option<(&FrameTimingContext, SpanId)>,
    ) -> Result<FrameInferenceOutput> {
        self.infer_input_with_spans(&prepared.values, prepared.geometry, timing)
    }

    fn prepare_rgb8_for_provider(
        &self,
        source_width: u32,
        source_height: u32,
        pixels: &[u8],
    ) -> Result<PreparedFrameInput> {
        let model_size = self.image_size as u32;
        let prepared = prepare_rgb8(source_width, source_height, pixels, model_size)?;
        Ok(PreparedFrameInput::from(prepared))
    }

    fn infer_input_with_spans(
        &mut self,
        values: &[f32],
        geometry: LetterboxGeometry,
        timing: Option<(&FrameTimingContext, SpanId)>,
    ) -> Result<FrameInferenceOutput> {
        let engine_start = Instant::now();

        let inference_start = Instant::now();
        let inference_span = timing
            .and_then(|(context, parent)| context.start_child_span("model_execute", parent).ok());
        let session_result = run_session(&mut self.session, values, self.image_size);
        drop(inference_span);
        let (output_shape, output_values) =
            session_result.context("inference backend failed for RGB frame")?;
        let inference_ms = elapsed_ms(inference_start);
        ensure!(
            output_shape == self.expected_output_shape,
            "model output tensor shape changed at runtime: {output_shape:?}"
        );

        let decode_start = Instant::now();
        let decode_span = timing
            .and_then(|(context, parent)| context.start_child_span("decode_nms", parent).ok());
        let detections = decode_yolo_v8_output_with_coordinate_scale(
            &output_shape
                .iter()
                .map(|dimension| *dimension as usize)
                .collect::<Vec<_>>(),
            &output_values,
            &self.class_names,
            geometry,
            self.confidence,
            self.iou,
            self.max_detections,
            self.output_coordinate_scale,
        )
        .context("failed to decode model output for RGB frame")?;
        drop(decode_span);
        let decode_and_nms_ms = elapsed_ms(decode_start);

        Ok(FrameInferenceOutput {
            source_width_px: geometry.source_width,
            source_height_px: geometry.source_height,
            detections,
            timings: FrameInferenceTimings {
                preprocess_ms: 0.0,
                inference_ms,
                decode_and_nms_ms,
                engine_total_ms: elapsed_ms(engine_start),
            },
            ball_refinements: None,
        })
    }

    pub fn model_identity(&self) -> &ModelIdentity {
        &self.model_identity
    }

    pub fn inference_settings(&self) -> &InferenceSettings {
        &self.inference_settings
    }

    /// The policy requested by configuration; this does not assert the
    /// provider or device that executed each model node.
    pub fn requested_provider(&self) -> ProviderSelection {
        self.requested_provider
    }

    fn output_shape(&self) -> &[i64] {
        &self.expected_output_shape
    }
}

/// Inference and artifact settings shared by recorded-video replay.
#[derive(Debug, Clone)]
pub struct AuditOptions {
    pub onnx_path: PathBuf,
    pub source_checkpoint_path: PathBuf,
    pub input_path: PathBuf,
    pub provider: ProviderSelection,
    pub image_size: u32,
    pub confidence: f32,
    pub iou: f32,
    pub max_detections: usize,
    pub intra_op_threads: Option<usize>,
    pub warmup_runs: usize,
    /// Source image coordinate space. This describes the input image bytes;
    /// letterbox resizing is later reversed before detector positions publish.
    pub source_image_space: ImageSpace,
    /// Optional JSONL artifact; write time is tracked separately from inference.
    pub detection_batch_output: Option<BatchOutputOptions>,
    /// Optional diagnostic geometry pass. It requires detection-batch output
    /// so the raw-box midpoint geometry diagnostic is reproducible from retained input.
    pub calibration: Option<CalibrationAuditOptions>,
    /// Immutable camera optics and ball-center calibration settings used by
    /// replay scene output. Missing assets preserve visibility only.
    pub ball_localization_settings: BallLocalizationSettings,
    /// Optional paced scene-snapshot replay. It requires detection-batch
    /// output so tracks can always be reconstructed from raw observations.
    pub scene_snapshot_output: Option<SceneSnapshotOutputOptions>,
}

/// Schedule policy for saved-video replay. Real-time mode follows the source
/// presentation timeline and processes only the latest due frame when behind;
/// capacity mode processes every frame sequentially without pacing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoReplayMode {
    RealtimeLatestFrame,
    Capacity,
}

/// Limits and evidence outputs for a bounded saved-video audit.
#[derive(Debug, Clone)]
pub struct VideoReplayOptions {
    /// Model, provider, tracking, and output settings. `input_path` is the video.
    pub audit: AuditOptions,
    /// Ordinal among video streams, matching temporal annotation metadata.
    pub video_stream_ordinal: u32,
    pub ffprobe_executable: PathBuf,
    pub ffmpeg_executable: PathBuf,
    pub mode: VideoReplayMode,
    pub maximum_video_bytes: u64,
    pub maximum_probe_json_bytes: usize,
    pub maximum_frame_count: usize,
    pub maximum_pixel_count: u64,
    pub maximum_decoded_video_bytes: u64,
    pub subprocess_timeout: Duration,
    /// Optional bounded JSONL details. Aggregate reports and requested
    /// detection/scene artifacts are independent of this trace cap.
    pub details_jsonl_path: Option<PathBuf>,
    /// Optional bounded image-space cue-geometry observations for review and
    /// annotation scoring. This trace contains scheduled cue updates only.
    pub phase_cue_geometry_jsonl_path: Option<PathBuf>,
    pub maximum_detail_records: usize,
}

/// Explicit scene output and already validated tracking policy for replay.
#[derive(Debug, Clone)]
pub struct SceneSnapshotOutputOptions {
    pub destination: SceneSnapshotOutputDestination,
    pub tracking: TrackingConfigV1,
}

/// Recorded-video scene snapshots are emitted as JSONL artifacts.
#[derive(Debug, Clone)]
pub enum SceneSnapshotOutputDestination {
    JsonlFile { path: PathBuf },
}

/// Explicit identities and destination for per-frame contract output.
#[derive(Debug, Clone)]
pub struct BatchOutputOptions {
    pub jsonl_path: PathBuf,
    pub engine_instance_id: String,
    pub source_session_id: String,
}

/// Explicit calibration and applied-runtime identity for an audit replay.
#[derive(Debug, Clone)]
pub struct CalibrationAuditOptions {
    bundle_path: PathBuf,
    bundle_sha256: String,
    calibration: ValidatedTableCalibrationV1,
    applied_camera_profile_id: String,
    applied_calibration_revision: u64,
    applied_rectification_revision: u64,
}

impl CalibrationAuditOptions {
    /// Load a versioned calibration bundle and bind its hash to the validated
    /// calibration used by this replay. Callers cannot independently replace
    /// the geometry or provenance after construction.
    pub fn load(
        bundle_path: impl Into<PathBuf>,
        applied_camera_profile_id: impl Into<String>,
        applied_calibration_revision: u64,
        applied_rectification_revision: u64,
    ) -> Result<Self> {
        let bundle_path = bundle_path.into();
        let bundle_bytes = fs::read(&bundle_path).with_context(|| {
            format!(
                "failed to read calibration bundle {}",
                bundle_path.display()
            )
        })?;
        let calibration = serde_json::from_slice::<TableCalibrationV1>(&bundle_bytes)
            .context("calibration bundle does not match the table-calibration v1 structure")?
            .into_validated()
            .context("calibration bundle failed structural or numerical validation")?;
        let applied_camera_profile_id = applied_camera_profile_id.into();
        validate_batch_identifier(&applied_camera_profile_id, "applied camera profile ID")?;
        ensure!(
            applied_calibration_revision <= MAX_SAFE_JSON_INTEGER
                && applied_rectification_revision <= MAX_SAFE_JSON_INTEGER,
            "applied calibration revisions must be safe JSON integers"
        );

        Ok(Self {
            bundle_path,
            bundle_sha256: format!("{:x}", Sha256::digest(&bundle_bytes)),
            calibration,
            applied_camera_profile_id,
            applied_calibration_revision,
            applied_rectification_revision,
        })
    }

    pub fn bundle_sha256(&self) -> &str {
        &self.bundle_sha256
    }

    pub fn calibration_revision(&self) -> u64 {
        self.calibration.calibration_revision()
    }

    pub fn applied_camera_profile_id(&self) -> &str {
        &self.applied_camera_profile_id
    }

    pub fn applied_rectification_revision(&self) -> u64 {
        self.applied_rectification_revision
    }
}

/// Explicit inference backend policy for live and recorded-video inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderSelection {
    /// Disable automatic accelerators and register the portable CPU provider.
    Cpu,
    /// Let ONNX Runtime choose its max-efficiency device policy.
    Auto,
    /// Explicitly request the native OpenVINO Runtime GPU backend.
    #[serde(rename = "openvino")]
    Openvino,
}

impl ProviderSelection {
    fn requested_provider_name(self) -> &'static str {
        match self {
            Self::Cpu => CPU::NAME,
            Self::Auto => "AutoDevicePolicy::MaxEfficiency",
            Self::Openvino => "OpenVINO GPU (native)",
        }
    }

    fn binding(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Openvino => (
                "openvino-rs",
                "0.11.1",
                "Native OpenVINO GPU backend explicitly selected; runtime EXECUTION_DEVICES is checked to confirm GPU execution, and per-node assignment is unavailable",
            ),
            Self::Cpu | Self::Auto => (
                "ort",
                "2.0.0-rc.15",
                "ONNX Runtime per-node profile from a separate identically configured session; empty observations mean unavailable",
            ),
        }
    }
}

/// Versioned evidence from a predecoded recorded-video run. Replay metrics
/// explicitly distinguish media presentation time from local monotonic time.
#[derive(Debug, Serialize)]
pub struct VideoReplayReport {
    pub schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub workload: &'static str,
    pub environment: crate::performance_audit_runner::RunEnvironment,
    pub mode: VideoReplayMode,
    pub source: VideoReplaySourceReport,
    pub model: VideoReplayModelReport,
    pub runtime: VideoReplayRuntimeReport,
    pub settings: VideoReplaySettingsReport,
    pub measurement: VideoReplayMeasurementReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_schedule: Option<VideoReplayPhaseReport>,
    pub artifacts: VideoReplayArtifactsReport,
    pub effective_tracking_configuration: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationAuditReport>,
    pub batch_engine_instance_id: Option<String>,
    pub source_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal_accuracy: Option<Value>,
    pub unsupported_measurements: Vec<&'static str>,
}

/// Source-timeline diagnostics for the shared phase scheduler used by replay.
/// Rates are computed against normalized media duration, independently from
/// wall-clock inference throughput in `measurement`.
#[derive(Debug, Serialize)]
pub struct VideoReplayPhaseReport {
    pub clock: &'static str,
    pub source_frames_considered: usize,
    pub motion_checks: usize,
    pub cue_observations: usize,
    pub detector_calls: usize,
    pub intentional_detector_skips: usize,
    /// Synchronous offline replay has no worker contention, but emits the
    /// field so the live and replay work counters share one vocabulary.
    pub overload_skips: usize,
    pub obsolete_results: usize,
    pub trusted_layout_duration_ms: f64,
    pub recovery_delays_ms: Vec<u64>,
    pub achieved_motion_checks_hz: Option<f64>,
    pub achieved_cue_observations_hz: Option<f64>,
    pub achieved_detector_calls_hz: Option<f64>,
    pub effective_schedule: Value,
}

#[derive(Debug, Serialize)]
pub struct VideoReplaySourceReport {
    pub path: String,
    pub sha256: String,
    pub video_stream_ordinal: u32,
    pub ffprobe_global_stream_index: u32,
    pub width_px: u32,
    pub height_px: u32,
    pub source_frame_count: usize,
    pub first_source_pts_seconds: f64,
    pub source_duration_seconds: Option<f64>,
    pub presentation_time_source: &'static str,
    pub ffprobe_version: String,
    pub ffmpeg_version: String,
    pub ffprobe_elapsed_ms: f64,
    pub decoded_raw_rgb_bytes: u64,
    pub offline_decode_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct VideoReplayModelReport {
    pub onnx_path: String,
    pub onnx_sha256: String,
    pub source_checkpoint_path: String,
    pub source_checkpoint_sha256: String,
    pub manifest_path: String,
    pub manifest_sha256: String,
    pub class_names: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct VideoReplayRuntimeReport {
    pub binding: &'static str,
    pub binding_version: &'static str,
    pub requested_execution_provider: &'static str,
    pub observed_execution_providers: Vec<String>,
    pub provider_event_count: usize,
    pub provider_node_counts: BTreeMap<String, usize>,
    pub provider_probe_session_init_ms: f64,
    pub provider_probe_inference_ms: f64,
    pub measurement_session_init_ms: f64,
    pub warmup_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct VideoReplaySettingsReport {
    pub image_size: u32,
    pub confidence: f32,
    pub iou: f32,
    pub max_detections: usize,
    pub intra_op_threads: Option<usize>,
    pub warmup_runs: usize,
    pub source_image_space: ImageSpace,
    pub maximum_video_bytes: u64,
    pub maximum_probe_json_bytes: usize,
    pub maximum_frame_count: usize,
    pub maximum_pixel_count: u64,
    pub maximum_decoded_video_bytes: u64,
    pub subprocess_timeout_ms: u128,
    pub maximum_detail_records: usize,
}

#[derive(Debug, Serialize)]
pub struct VideoReplayMeasurementReport {
    pub scheduled_source_frames: usize,
    pub decoded_source_frames: usize,
    pub processed_source_frames: usize,
    pub latest_frame_replacements: usize,
    /// Processed observations completed after the next source-frame PTS.
    /// Capacity replay has no source deadline and reports this as unavailable.
    pub processed_source_deadline_misses: Option<usize>,
    pub source_media_duration_seconds: Option<f64>,
    pub measured_wall_ms: f64,
    pub mean_preprocess_ms: f64,
    pub mean_inference_ms: f64,
    pub mean_decode_and_nms_ms: f64,
    pub total_batch_export_ms: f64,
    pub total_scene_tracking_ms: f64,
    pub source_observations_per_second: Option<f64>,
    /// In real-time mode, successful results completed inside the source media
    /// window divided by that exact duration. Capacity mode uses wall time.
    pub processed_observations_per_second: Option<f64>,
    pub processed_completions_in_source_window: Option<usize>,
    pub processed_completions_after_source_window: Option<usize>,
    pub source_to_inference_start_ms: LatencyPercentiles,
    pub source_to_inference_complete_ms: LatencyPercentiles,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_to_calibration_complete_ms: Option<LatencyPercentiles>,
}

#[derive(Debug, Serialize)]
pub struct VideoReplayArtifactsReport {
    pub detection_batch_path: Option<String>,
    pub detection_batch_count: usize,
    pub scene_snapshot_path: Option<String>,
    pub scene_snapshot_count: usize,
    pub detailed_trace_path: Option<String>,
    pub detailed_trace_written: usize,
    pub detailed_trace_dropped_by_limit: usize,
    pub phase_cue_geometry_trace_path: Option<String>,
    pub phase_cue_geometry_trace_written: usize,
    pub phase_cue_geometry_trace_dropped_by_limit: usize,
    pub artifact_writes_included_in_measurement: bool,
    pub offline_decode_excluded_from_measurement: bool,
}

#[derive(Debug, Serialize)]
pub struct CalibrationAuditReport {
    pub bundle_path: String,
    pub bundle_sha256: String,
    pub bundle_calibration_revision: u64,
    pub bundle_camera_profile_id: String,
    pub bundle_rectification_revision: u64,
    pub bundle_image_space: CalibrationImageSpace,
    pub bundle_source_width_px: u32,
    pub bundle_source_height_px: u32,
    pub applied_calibration_revision: u64,
    pub applied_camera_profile_id: String,
    pub applied_rectification_revision: u64,
    pub applied_image_space: ImageSpace,
    pub frame_count: usize,
    pub calibrated_frame_count: usize,
    pub unavailable_frame_count: usize,
    pub calibrated_detection_count: usize,
}

#[derive(Debug, Serialize)]
pub struct FrameCalibrationReport {
    pub status: CalibrationStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Number of raw detector-box midpoints the table homography accepted for
    /// geometry diagnostics. These are not refined ball centers and are never
    /// emitted as physical ball positions.
    pub projected_box_midpoint_count: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    Valid,
    Unavailable,
}

type DetectionBatchSink = BufWriter<fs::File>;
#[derive(Debug, Serialize)]
pub struct LatencyPercentiles {
    pub sample_count: usize,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
struct FrameTimings {
    pub image_decode_ms: f64,
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub decode_and_nms_ms: f64,
    pub end_to_end_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ball_refinement_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_tracking_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_encode_enqueue_ms: Option<f64>,
}

struct FrameResult {
    timings: FrameTimings,
    source_to_inference_start_ms: Option<f64>,
    source_to_inference_complete_ms: Option<f64>,
    source_to_calibration_complete_ms: Option<f64>,
    detections: Vec<Detection>,
    calibration: Option<FrameCalibrationReport>,
}

#[derive(Debug, Deserialize)]
struct ExportManifest {
    manifest_schema_version: u32,
    source_model: SourceModelManifest,
    exported_model: ExportedModelManifest,
    model: ModelManifest,
}

#[derive(Debug, Deserialize)]
struct SourceModelManifest {
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct ExportedModelManifest {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct ModelManifest {
    class_count: usize,
    class_names: Vec<ClassManifest>,
}

#[derive(Debug, Deserialize)]
struct ClassManifest {
    id: usize,
    name: String,
}

#[derive(Debug, Clone)]
struct LoadedModelContract {
    manifest_path: PathBuf,
    manifest_sha256: String,
    artifact_sha256: String,
    source_checkpoint_sha256: String,
    class_names: Vec<String>,
    output_coordinate_scale: f32,
}

struct ProviderEvidence {
    observed_providers: Vec<String>,
    event_count: usize,
    node_counts: BTreeMap<String, usize>,
}

struct TimedInput {
    image: RgbImage,
    prepared: PreparedInput,
    decode_ms: f64,
    preprocess_ms: f64,
}

/// One immutable time interval shared by the replay source and inference
/// worker. The half-open boundary makes events at `end` unambiguously part of
/// drain rather than the measured source interval.
#[derive(Debug, Clone, Copy)]
struct PacedReplayMeasurementWindow {
    start: Instant,
    end: Instant,
}

impl PacedReplayMeasurementWindow {
    fn new(start: Instant, duration: Duration) -> Result<Self> {
        ensure!(
            !duration.is_zero(),
            "paced replay measurement window is empty"
        );
        let end = start
            .checked_add(duration)
            .context("paced replay measurement window exceeds the monotonic clock range")?;
        Ok(Self { start, end })
    }

    fn contains(self, event_at: Instant) -> bool {
        self.start <= event_at && event_at < self.end
    }

    fn duration(self) -> Duration {
        self.end.duration_since(self.start)
    }
}

fn frames_per_second(frame_count: usize, window: PacedReplayMeasurementWindow) -> f64 {
    frame_count as f64 / window.duration().as_secs_f64()
}

struct FrameToProcess {
    source_index: usize,
    path: PathBuf,
    timed_input: TimedInput,
    scheduled_at: Option<Instant>,
    source_arrived_at: Option<Instant>,
}

struct FrameProcessContext<'a> {
    inference_engine: &'a mut FrameInferenceEngine,
    class_names: &'a [String],
    options: &'a AuditOptions,
    localization_profile: Option<&'a LoadedBallLocalizationProfile>,
    onnx_sha256: &'a str,
    source_checkpoint_sha256: &'a str,
    batch_sink: &'a mut Option<DetectionBatchSink>,
    scene_output: &'a mut Option<SceneSnapshotOutputRuntime>,
    batch_clock_origin: Instant,
}

struct SceneSnapshotOutputRuntime {
    tracker: SceneTrackerV1,
    tracking: TrackingConfigV1,
    sink: SceneSnapshotSink,
    snapshot_count: usize,
    total_tracking_ms: f64,
}

enum SceneSnapshotSink {
    Jsonl(BufWriter<fs::File>),
}

/// Run the current Rust detector against one selected video stream. Video is
/// decoded once into a bounded temporary raw-RGB spool before the measured
/// interval. This keeps ffmpeg startup and offline decode separate from the
/// inference schedule, while preserving source PTS for the realtime mode.
/// The temporary file is removed automatically on every return path.
pub fn run_video_audit(options: &VideoReplayOptions) -> Result<VideoReplayReport> {
    run_video_audit_with_schedule(options, Some(SchedulerConfig::default()))
}

/// Run the recorded-video coordinator with an optional deterministic phase
/// schedule. Scheduler time is derived only from normalized source PTS, so
/// capacity-mode execution speed cannot change dispatch cadence or settlement.
pub fn run_video_audit_with_schedule(
    options: &VideoReplayOptions,
    scheduler_config: Option<SchedulerConfig>,
) -> Result<VideoReplayReport> {
    let audit = &options.audit;
    if let Some(config) = scheduler_config.as_ref() {
        config
            .validate()
            .context("invalid recorded-video phase schedule")?;
    }
    validate_video_audit_options(audit)?;
    ensure!(
        options.maximum_video_bytes > 0
            && options.maximum_probe_json_bytes > 0
            && options.maximum_frame_count > 0
            && options.maximum_pixel_count > 0
            && options.maximum_decoded_video_bytes > 0
            && !options.subprocess_timeout.is_zero(),
        "video replay resource limits and subprocess timeout must be positive"
    );
    ensure!(
        options.details_jsonl_path.is_some()
            || options.phase_cue_geometry_jsonl_path.is_some()
            || options.maximum_detail_records == 0,
        "a nonzero detail-record limit requires a detail or cue-geometry JSONL destination"
    );

    let video_path = &audit.input_path;
    let video_metadata = fs::metadata(video_path)
        .with_context(|| format!("video does not exist: {}", video_path.display()))?;
    ensure!(
        video_metadata.is_file(),
        "video input is not a regular file: {}",
        video_path.display()
    );
    ensure!(
        video_metadata.len() <= options.maximum_video_bytes,
        "video is {} bytes, exceeding configured limit {}",
        video_metadata.len(),
        options.maximum_video_bytes
    );

    let ffprobe_version =
        executable_version(&options.ffprobe_executable, options.subprocess_timeout)?;
    let ffmpeg_version =
        executable_version(&options.ffmpeg_executable, options.subprocess_timeout)?;
    let ffprobe_global_stream_index = probe_global_video_stream_index(
        &options.ffprobe_executable,
        video_path,
        options.video_stream_ordinal,
        options.subprocess_timeout,
    )?;
    let timeline_limits = VideoTimelineLimits {
        maximum_probe_json_bytes: options.maximum_probe_json_bytes,
        maximum_frame_count: options.maximum_frame_count,
        maximum_pixel_count: options.maximum_pixel_count,
    };
    let probe_started = Instant::now();
    let probe_json = probe_video_timeline(
        &options.ffprobe_executable,
        video_path,
        options.video_stream_ordinal,
        timeline_limits.maximum_probe_json_bytes,
        options.subprocess_timeout,
    )?;
    let timeline =
        parse_ffprobe_timeline(&probe_json, ffprobe_global_stream_index, timeline_limits)?;
    let ffprobe_elapsed_ms = elapsed_ms(probe_started);
    ensure!(
        timeline.frame_count() <= options.maximum_frame_count,
        "video frame count exceeds configured limit"
    );
    let frame_bytes = usize::try_from(timeline.width)
        .ok()
        .and_then(|width| {
            usize::try_from(timeline.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(3))
        .context("decoded RGB frame allocation size overflowed")?;
    let frame_pixels = u64::from(timeline.width)
        .checked_mul(u64::from(timeline.height))
        .context("decoded pixel count overflowed")?;
    ensure!(
        frame_pixels <= options.maximum_pixel_count,
        "decoded video geometry exceeds configured pixel limit"
    );
    ensure!(
        frame_bytes > 0,
        "decoded video frame has an empty RGB allocation"
    );
    let expected_decoded_bytes = u64::try_from(frame_bytes)
        .context("decoded frame byte count exceeds u64")?
        .checked_mul(
            u64::try_from(timeline.frame_count()).context("video frame count exceeds u64")?,
        )
        .context("decoded video byte count overflowed")?;
    ensure!(
        expected_decoded_bytes <= options.maximum_decoded_video_bytes,
        "decoded raw RGB spool requires {expected_decoded_bytes} bytes, exceeding configured limit {}",
        options.maximum_decoded_video_bytes
    );

    let source_video_sha256 = sha256_file_bounded(video_path, options.maximum_video_bytes)?;
    let (mut raw_frames, offline_decode_ms, decoded_raw_rgb_bytes) = decode_video_to_spool(
        &options.ffmpeg_executable,
        video_path,
        timeline.stream_index,
        expected_decoded_bytes,
        options.maximum_decoded_video_bytes,
        options.subprocess_timeout,
    )?;
    ensure!(
        decoded_raw_rgb_bytes == expected_decoded_bytes,
        "ffmpeg decoded {decoded_raw_rgb_bytes} raw RGB bytes; expected {expected_decoded_bytes} bytes for {} frames at {}x{}",
        timeline.frame_count(),
        timeline.width,
        timeline.height
    );

    let first_image = read_spooled_rgb_frame(
        &mut raw_frames,
        0,
        frame_bytes,
        timeline.width,
        timeline.height,
    )?;
    let first_prepared = prepare_decoded_image(first_image, 0.0, audit.image_size)?.prepared;
    let manifest = load_and_verify_manifest(&audit.onnx_path, &audit.source_checkpoint_path)?;
    validate_scene_classes(audit.scene_snapshot_output.as_ref(), &manifest.class_names)?;
    let expected_output_shape = vec![
        1_i64,
        (4 + manifest.class_names.len()) as i64,
        expected_anchor_count(audit.image_size)? as i64,
    ];
    let environment = ort::init()
        .with_name("billiards-perception-recorded-video-audit")
        .with_telemetry(false)
        .build()
        .context("failed to initialize ONNX Runtime")?;
    let (
        profile_evidence,
        provider_probe_session_init_ms,
        provider_probe_inference_ms,
        first_prepared,
    ) = profile_provider(
        &environment,
        audit,
        &manifest.class_names,
        &expected_output_shape,
        first_prepared,
    )?;
    let (mut inference_engine, measurement_session_init_ms) =
        FrameInferenceEngine::from_verified_model(
            environment.clone(),
            FrameInferenceOptions {
                onnx_path: Some(audit.onnx_path.clone()),
                source_checkpoint_path: audit.source_checkpoint_path.clone(),
                provider: audit.provider,
                image_size: audit.image_size,
                confidence: audit.confidence,
                iou: audit.iou,
                max_detections: audit.max_detections,
                intra_op_threads: audit.intra_op_threads,
            },
            manifest.clone(),
        )?;
    ensure!(
        inference_engine.output_shape() == expected_output_shape,
        "measurement output signature changed after provider probe: {:?}",
        inference_engine.output_shape()
    );
    let warmup_started = Instant::now();
    for _ in 0..audit.warmup_runs {
        let _ = inference_engine.infer_prepared(&first_prepared)?;
    }
    let warmup_ms = elapsed_ms(warmup_started);

    let mut batch_sink = audit
        .detection_batch_output
        .as_ref()
        .map(open_detection_batch_sink)
        .transpose()?;
    let mut scene_output = audit
        .scene_snapshot_output
        .as_ref()
        .map(open_scene_snapshot_output)
        .transpose()?;
    let localization_profile = if let Some(scene_settings) = audit.scene_snapshot_output.as_ref() {
        let mut settings = audit.ball_localization_settings.clone();
        settings.validate()?;
        let project_root =
            std::env::current_dir().context("could not resolve replay localization asset paths")?;
        settings.resolve_paths(&project_root);
        match LoadedBallLocalizationProfile::load_for_frame(
            settings,
            scene_settings.tracking.camera_profile_id(),
            match audit.source_image_space {
                ImageSpace::Raw => crate::ImageSpace::Raw,
                ImageSpace::Rectified => crate::ImageSpace::Rectified,
            },
            timeline.width,
            timeline.height,
            scene_settings.tracking.rectification_revision(),
        ) {
            Ok(profile) => {
                if let Some(error) = profile.calibration_error.as_deref() {
                    eprintln!(
                        "replay ball-center calibration unavailable; precise table positions are suppressed: {error}"
                    );
                }
                Some(profile)
            }
            Err(error) => {
                eprintln!(
                    "replay ball-localization profile unavailable; precise table positions are suppressed: {error:#}"
                );
                None
            }
        }
    } else {
        None
    };
    let mut detail_writer = options
        .details_jsonl_path
        .as_ref()
        .map(|path| create_new_buffered_file(path, "video detail trace"))
        .transpose()?;
    let mut cue_geometry_writer = options
        .phase_cue_geometry_jsonl_path
        .as_ref()
        .map(|path| create_new_buffered_file(path, "phase cue geometry trace"))
        .transpose()?;

    let measurement_started = Instant::now();
    let source_duration_seconds = timeline_media_duration(&timeline);
    let measurement_window = source_duration_seconds
        .map(checked_duration_from_seconds)
        .transpose()?
        .filter(|duration| !duration.is_zero())
        .map(|duration| PacedReplayMeasurementWindow::new(measurement_started, duration))
        .transpose()?;
    let mut context = FrameProcessContext {
        inference_engine: &mut inference_engine,
        class_names: &manifest.class_names,
        options: audit,
        localization_profile: localization_profile.as_ref(),
        onnx_sha256: &manifest.artifact_sha256,
        source_checkpoint_sha256: &manifest.source_checkpoint_sha256,
        batch_sink: &mut batch_sink,
        scene_output: &mut scene_output,
        batch_clock_origin: measurement_started,
    };
    let effective_scheduler_config = scheduler_config.clone();
    let mut phase_scheduler = scheduler_config
        .map(PhaseScheduler::new)
        .transpose()
        .context("could not initialize recorded-video phase scheduler")?;
    let mut motion_monitor = phase_scheduler
        .as_ref()
        .map(|_| MotionMonitor::new(MotionMonitorConfig::default()))
        .transpose()
        .context("could not initialize recorded-video motion monitor")?;
    let cue_geometry_config = CueGeometryConfig::default();
    let mut cue_proposals = Vec::<CueProposal>::new();
    let mut phase_source_frames = 0usize;
    let mut phase_motion_checks = 0usize;
    let mut phase_cue_observations = 0usize;
    let mut phase_detector_calls = 0usize;
    let mut phase_intentional_skips = 0usize;
    let mut trusted_layout_duration = Duration::ZERO;
    let mut recovery_started = None;
    let mut recovery_delays_ms = Vec::new();
    let mut previous_scheduler_time = None;
    let mut previous_layout_trusted = false;
    let mut next_source_index = 0usize;
    let mut processed_source_frames = 0usize;
    let mut latest_frame_replacements = 0usize;
    let mut processed_source_deadline_misses = 0usize;
    let mut completions_in_source_window = 0usize;
    let mut completions_after_source_window = 0usize;
    let mut source_to_start_samples = Vec::with_capacity(timeline.frame_count().min(4096));
    let mut source_to_complete_samples = Vec::with_capacity(timeline.frame_count().min(4096));
    let mut source_to_calibration_samples = Vec::with_capacity(timeline.frame_count().min(4096));
    let mut calibrated_frame_count = 0usize;
    let mut calibrated_detection_count = 0usize;
    let mut preprocess_total_ms = 0.0;
    let mut inference_total_ms = 0.0;
    let mut decode_and_nms_total_ms = 0.0;
    let mut batch_export_total_ms = 0.0;
    let mut trace_written = 0usize;
    let mut trace_limited = 0usize;
    let mut cue_geometry_trace_written = 0usize;
    let mut cue_geometry_trace_limited = 0usize;
    let mut last_progress = measurement_started;

    while next_source_index < timeline.frame_count() {
        let source_index = match options.mode {
            VideoReplayMode::RealtimeLatestFrame => {
                let candidate_deadline = video_frame_deadline(
                    measurement_started,
                    timeline.presentation_times_seconds[next_source_index],
                )?;
                wait_until(candidate_deadline);
                let elapsed_source_seconds = measurement_started.elapsed().as_secs_f64();
                let latest_due = latest_due_source_index(
                    &timeline.presentation_times_seconds,
                    next_source_index,
                    elapsed_source_seconds,
                );
                latest_frame_replacements += latest_due - next_source_index;
                latest_due
            }
            VideoReplayMode::Capacity => next_source_index,
        };
        let source_pts_seconds = timeline.presentation_times_seconds[source_index];
        let source_scheduled_at = if options.mode == VideoReplayMode::RealtimeLatestFrame {
            Some(video_frame_deadline(
                measurement_started,
                source_pts_seconds,
            )?)
        } else {
            None
        };
        let image = read_spooled_rgb_frame(
            &mut raw_frames,
            source_index,
            frame_bytes,
            timeline.width,
            timeline.height,
        )?;
        let source_time = checked_duration_from_seconds(source_pts_seconds)?;
        let mut run_detector = phase_scheduler.is_none();
        let mut cue_geometry_observation = None;
        if let Some(scheduler) = phase_scheduler.as_mut() {
            phase_source_frames = phase_source_frames.saturating_add(1);
            if let Some(previous_time) = previous_scheduler_time {
                if previous_layout_trusted {
                    trusted_layout_duration = trusted_layout_duration
                        .saturating_add(source_time.saturating_sub(previous_time));
                }
            }
            let phase_before = scheduler.phase();
            let mut decision = scheduler
                .tick(
                    source_time,
                    crate::phase_scheduler::WorkAvailability::default(),
                )
                .context("recorded-video scheduler rejected source presentation time")?;
            if decision.run_motion {
                let assessment = motion_monitor
                    .as_mut()
                    .expect("phase scheduler and motion monitor are initialized together")
                    .observe_rgb8(source_time, timeline.width, timeline.height, image.as_raw())
                    .context("recorded-video motion assessment failed")?;
                phase_motion_checks = phase_motion_checks.saturating_add(1);
                let motion_evidence = match assessment.classification {
                    MotionClassification::Quiet => MotionEvidence::Quiet,
                    MotionClassification::Moving => MotionEvidence::Moving,
                    MotionClassification::OccludedOrUncertain => {
                        MotionEvidence::OccludedOrUncertain
                    }
                };
                scheduler
                    .observe_motion(source_time, motion_evidence)
                    .context("recorded-video scheduler rejected motion evidence")?;
            }
            if scheduler.phase() != phase_before {
                decision = scheduler
                    .tick(
                        source_time,
                        crate::phase_scheduler::WorkAvailability::default(),
                    )
                    .context("recorded-video scheduler rejected phase transition")?;
                decision.run_motion = false;
            }
            let mut evidence = scheduler
                .evidence(source_time)
                .context("recorded-video scheduler evidence is unavailable")?;
            if evidence.phase == Phase::Shot {
                recovery_started.get_or_insert(source_time);
            }
            if decision.run_cue && evidence.phase != Phase::Shot {
                let mut cue = estimate_cue_geometry(
                    image.as_raw(),
                    timeline.width,
                    timeline.height,
                    &cue_proposals,
                    Duration::ZERO,
                    cue_geometry_config,
                )
                .map_err(|error| {
                    anyhow::anyhow!("recorded-video cue geometry assessment failed: {error:?}")
                })?;
                if let (Some(geometry), Some(calibration)) =
                    (cue.as_mut(), audit.calibration.as_ref())
                {
                    let context = CalibrationContext {
                        calibration_revision: calibration.applied_calibration_revision,
                        camera_profile_id: &calibration.applied_camera_profile_id,
                        rectification_revision: calibration.applied_rectification_revision,
                        image_space: match audit.source_image_space {
                            ImageSpace::Raw => crate::ImageSpace::Raw,
                            ImageSpace::Rectified => crate::ImageSpace::Rectified,
                        },
                        source_width_px: timeline.width,
                        source_height_px: timeline.height,
                    };
                    orient_cue_geometry_from_playing_surface(
                        geometry,
                        &calibration.calibration,
                        context,
                    );
                }
                cue_geometry_observation = Some(cue);
                phase_cue_observations = phase_cue_observations.saturating_add(1);
                scheduler
                    .observe_cue(
                        source_time,
                        if cue.is_some() {
                            CueEvidence::Visible
                        } else {
                            CueEvidence::NotVisible
                        },
                    )
                    .context("recorded-video scheduler rejected cue evidence")?;
                evidence = scheduler
                    .evidence(source_time)
                    .context("recorded-video scheduler evidence is unavailable after cue update")?;
            }
            // Ball observations remain useful while balls are moving. The
            // scheduler still invalidates verified layout/guidance immediately
            // on motion, and only quiet dwell plus consensus can restore trust.
            run_detector = decision.run_ball;
            if run_detector {
                phase_detector_calls = phase_detector_calls.saturating_add(1);
            } else {
                phase_intentional_skips = phase_intentional_skips.saturating_add(1);
            }
            let layout_trusted = evidence.map_trusted;
            if previous_layout_trusted && !layout_trusted {
                recovery_started = Some(source_time);
            } else if !previous_layout_trusted && layout_trusted {
                if let Some(started) = recovery_started.take() {
                    recovery_delays_ms.push(
                        source_time
                            .saturating_sub(started)
                            .as_millis()
                            .min(u128::from(u64::MAX)) as u64,
                    );
                }
            }
            previous_layout_trusted = layout_trusted;
            previous_scheduler_time = Some(source_time);
        }
        if !run_detector {
            if let Some(cue) = cue_geometry_observation {
                if let Some(writer) = cue_geometry_writer.as_mut() {
                    if cue_geometry_trace_written < options.maximum_detail_records {
                        write_cue_geometry_trace(
                            writer,
                            &source_video_sha256,
                            source_index,
                            source_pts_seconds,
                            cue,
                        )?;
                        cue_geometry_trace_written += 1;
                    } else {
                        cue_geometry_trace_limited += 1;
                    }
                }
            }
            next_source_index = source_index + 1;
            if last_progress.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "video audit: mode={:?} selected={phase_source_frames}/{} detector_calls={phase_detector_calls} replaced={latest_frame_replacements} source_elapsed_ms={:.0}",
                    options.mode,
                    timeline.frame_count(),
                    measurement_started.elapsed().as_secs_f64() * 1000.0
                );
                last_progress = Instant::now();
            }
            continue;
        }
        let source_arrived_at = source_scheduled_at.or_else(|| Some(Instant::now()));
        let timed_input = prepare_decoded_image(image, 0.0, audit.image_size)?;
        let (result, export_ms, inference_completed_at) = process_frame(
            &mut context,
            FrameToProcess {
                source_index,
                path: video_path.clone(),
                timed_input,
                scheduled_at: source_scheduled_at,
                source_arrived_at,
            },
            processed_source_frames + 1,
        )?;
        if let Some(scheduler) = phase_scheduler.as_mut() {
            let was_trusted = scheduler.map_trusted();
            let ball_positions = result
                .detections
                .iter()
                .filter(|detection| detection.class_name.eq_ignore_ascii_case("ball"))
                .map(|detection| ImagePoint {
                    x: f64::from(detection.center_xy_px[0]),
                    y: f64::from(detection.center_xy_px[1]),
                })
                .collect();
            scheduler
                .observe_layout(source_time, LayoutObservation::Success(ball_positions))
                .context("recorded-video scheduler rejected detector result")?;
            cue_proposals = replay_cue_proposals(
                &result.detections,
                timeline.width,
                timeline.height,
                cue_geometry_config.max_proposals,
            );
            let is_trusted = scheduler.map_trusted();
            if !was_trusted && is_trusted {
                motion_monitor
                    .as_mut()
                    .expect("phase scheduler and motion monitor are initialized together")
                    .set_current_as_verified_reference()
                    .context("failed to promote verified replay layout motion reference")?;
                if let Some(started) = recovery_started.take() {
                    recovery_delays_ms.push(
                        source_time
                            .saturating_sub(started)
                            .as_millis()
                            .min(u128::from(u64::MAX)) as u64,
                    );
                }
            } else if was_trusted && !is_trusted {
                recovery_started = Some(source_time);
            }
            previous_layout_trusted = is_trusted;
        }
        if options.mode == VideoReplayMode::RealtimeLatestFrame {
            if let Some(next_pts) = timeline
                .presentation_times_seconds
                .get(source_index.saturating_add(1))
            {
                if completed_after_source_deadline(
                    measurement_started,
                    *next_pts,
                    inference_completed_at,
                )? {
                    processed_source_deadline_misses += 1;
                }
            }
        }
        if let Some(window) = measurement_window {
            if window.contains(inference_completed_at) {
                completions_in_source_window += 1;
            } else if inference_completed_at >= window.end {
                completions_after_source_window += 1;
            }
        }
        if let Some(latency) = result.source_to_inference_start_ms {
            source_to_start_samples.push(latency);
        }
        if let Some(latency) = result.source_to_inference_complete_ms {
            source_to_complete_samples.push(latency);
        }
        if let Some(latency) = result.source_to_calibration_complete_ms {
            source_to_calibration_samples.push(latency);
        }
        if let Some(calibration) = &result.calibration {
            if matches!(calibration.status, CalibrationStatus::Valid) {
                calibrated_frame_count += 1;
                calibrated_detection_count +=
                    calibration.projected_box_midpoint_count.unwrap_or_default();
            }
        }
        preprocess_total_ms += result.timings.preprocess_ms;
        inference_total_ms += result.timings.inference_ms;
        decode_and_nms_total_ms += result.timings.decode_and_nms_ms;
        batch_export_total_ms += export_ms;
        if let Some(writer) = detail_writer.as_mut() {
            if trace_written < options.maximum_detail_records {
                write_video_detail(writer, source_index, source_pts_seconds, &result)?;
                trace_written += 1;
            } else {
                trace_limited += 1;
            }
        }
        if let Some(cue) = cue_geometry_observation {
            if let Some(writer) = cue_geometry_writer.as_mut() {
                if cue_geometry_trace_written < options.maximum_detail_records {
                    write_cue_geometry_trace(
                        writer,
                        &source_video_sha256,
                        source_index,
                        source_pts_seconds,
                        cue,
                    )?;
                    cue_geometry_trace_written += 1;
                } else {
                    cue_geometry_trace_limited += 1;
                }
            }
        }
        processed_source_frames += 1;
        next_source_index = source_index + 1;
        if last_progress.elapsed() >= Duration::from_secs(1) {
            eprintln!(
                "video audit: mode={:?} processed={processed_source_frames}/{} replaced={latest_frame_replacements} source_elapsed_ms={:.0}",
                options.mode,
                timeline.frame_count(),
                measurement_started.elapsed().as_secs_f64() * 1000.0
            );
            last_progress = Instant::now();
        }
    }
    if previous_layout_trusted {
        if let (Some(last_time), Some(duration_seconds)) =
            (previous_scheduler_time, source_duration_seconds)
        {
            let media_end = checked_duration_from_seconds(duration_seconds)?;
            trusted_layout_duration =
                trusted_layout_duration.saturating_add(media_end.saturating_sub(last_time));
        }
    }
    if let Some(writer) = detail_writer.as_mut() {
        writer
            .flush()
            .context("failed to flush video detail trace")?;
    }
    if let Some(writer) = cue_geometry_writer.as_mut() {
        writer
            .flush()
            .context("failed to flush phase cue geometry trace")?;
    }
    let measured_wall_ms = elapsed_ms(measurement_started);
    let processed_observations_per_second = match options.mode {
        VideoReplayMode::RealtimeLatestFrame => {
            measurement_window.map(|window| frames_per_second(completions_in_source_window, window))
        }
        VideoReplayMode::Capacity => {
            let wall_seconds = measured_wall_ms / 1000.0;
            (wall_seconds > 0.0).then_some(processed_source_frames as f64 / wall_seconds)
        }
    };
    let source_observations_per_second = match options.mode {
        VideoReplayMode::RealtimeLatestFrame => {
            measurement_window.map(|window| frames_per_second(timeline.frame_count(), window))
        }
        VideoReplayMode::Capacity => None,
    };
    let scene_snapshot_count = scene_output
        .as_ref()
        .map(|scene_runtime| scene_runtime.snapshot_count)
        .unwrap_or(0);
    let report = VideoReplayReport {
        schema_version: 1,
        run_id: None,
        workload: match options.mode {
            VideoReplayMode::RealtimeLatestFrame => {
                "recorded_video_pts_paced_latest_frame_predecoded"
            }
            VideoReplayMode::Capacity => "recorded_video_unpaced_capacity_predecoded",
        },
        environment: crate::performance_audit_runner::collect_environment(),
        mode: options.mode,
        source: VideoReplaySourceReport {
            path: video_path.display().to_string(),
            sha256: source_video_sha256,
            video_stream_ordinal: options.video_stream_ordinal,
            ffprobe_global_stream_index,
            width_px: timeline.width,
            height_px: timeline.height,
            source_frame_count: timeline.frame_count(),
            first_source_pts_seconds: timeline.origin_seconds,
            source_duration_seconds,
            presentation_time_source:
                "ffprobe_best_effort_timestamp_time_normalized_to_first_frame",
            ffprobe_version,
            ffmpeg_version,
            ffprobe_elapsed_ms,
            decoded_raw_rgb_bytes,
            offline_decode_ms,
        },
        model: VideoReplayModelReport {
            onnx_path: audit.onnx_path.display().to_string(),
            onnx_sha256: manifest.artifact_sha256,
            source_checkpoint_path: audit.source_checkpoint_path.display().to_string(),
            source_checkpoint_sha256: manifest.source_checkpoint_sha256,
            manifest_path: manifest.manifest_path.display().to_string(),
            manifest_sha256: manifest.manifest_sha256,
            class_names: manifest.class_names,
        },
        runtime: VideoReplayRuntimeReport {
            binding: audit.provider.binding().0,
            binding_version: audit.provider.binding().1,
            requested_execution_provider: audit.provider.requested_provider_name(),
            observed_execution_providers: profile_evidence.observed_providers,
            provider_event_count: profile_evidence.event_count,
            provider_node_counts: profile_evidence.node_counts,
            provider_probe_session_init_ms,
            provider_probe_inference_ms,
            measurement_session_init_ms,
            warmup_ms,
        },
        settings: VideoReplaySettingsReport {
            image_size: audit.image_size,
            confidence: audit.confidence,
            iou: audit.iou,
            max_detections: audit.max_detections,
            intra_op_threads: audit
                .intra_op_threads
                .filter(|_| audit.provider != ProviderSelection::Openvino),
            warmup_runs: audit.warmup_runs,
            source_image_space: audit.source_image_space,
            maximum_video_bytes: options.maximum_video_bytes,
            maximum_probe_json_bytes: options.maximum_probe_json_bytes,
            maximum_frame_count: options.maximum_frame_count,
            maximum_pixel_count: options.maximum_pixel_count,
            maximum_decoded_video_bytes: options.maximum_decoded_video_bytes,
            subprocess_timeout_ms: options.subprocess_timeout.as_millis(),
            maximum_detail_records: options.maximum_detail_records,
        },
        measurement: VideoReplayMeasurementReport {
            scheduled_source_frames: timeline.frame_count(),
            decoded_source_frames: timeline.frame_count(),
            processed_source_frames,
            latest_frame_replacements,
            processed_source_deadline_misses: (options.mode
                == VideoReplayMode::RealtimeLatestFrame)
                .then_some(processed_source_deadline_misses),
            source_media_duration_seconds: source_duration_seconds,
            measured_wall_ms,
            mean_preprocess_ms: mean_or_zero(preprocess_total_ms, processed_source_frames),
            mean_inference_ms: mean_or_zero(inference_total_ms, processed_source_frames),
            mean_decode_and_nms_ms: mean_or_zero(decode_and_nms_total_ms, processed_source_frames),
            total_batch_export_ms: batch_export_total_ms,
            total_scene_tracking_ms: scene_output
                .as_ref()
                .map(|scene_runtime| scene_runtime.total_tracking_ms)
                .unwrap_or(0.0),
            source_observations_per_second,
            processed_observations_per_second,
            processed_completions_in_source_window: measurement_window
                .map(|_| completions_in_source_window),
            processed_completions_after_source_window: measurement_window
                .map(|_| completions_after_source_window),
            source_to_inference_start_ms: percentile_summary(source_to_start_samples),
            source_to_inference_complete_ms: percentile_summary(source_to_complete_samples),
            source_to_calibration_complete_ms: calibration_latency_summary(
                options.mode,
                audit.calibration.is_some(),
                source_to_calibration_samples,
            ),
        },
        phase_schedule: effective_scheduler_config.as_ref().map(|config| {
            let media_duration = source_duration_seconds.filter(|duration| *duration > 0.0);
            VideoReplayPhaseReport {
                clock: "normalized_source_presentation_timestamps",
                source_frames_considered: phase_source_frames,
                motion_checks: phase_motion_checks,
                cue_observations: phase_cue_observations,
                detector_calls: phase_detector_calls,
                intentional_detector_skips: phase_intentional_skips,
                overload_skips: 0,
                obsolete_results: 0,
                trusted_layout_duration_ms: trusted_layout_duration.as_secs_f64() * 1000.0,
                recovery_delays_ms,
                achieved_motion_checks_hz: media_duration
                    .map(|duration| phase_motion_checks as f64 / duration),
                achieved_cue_observations_hz: media_duration
                    .map(|duration| phase_cue_observations as f64 / duration),
                achieved_detector_calls_hz: media_duration
                    .map(|duration| phase_detector_calls as f64 / duration),
                effective_schedule: replay_schedule_value(config),
            }
        }),
        artifacts: VideoReplayArtifactsReport {
            detection_batch_path: audit
                .detection_batch_output
                .as_ref()
                .map(|output| output.jsonl_path.display().to_string()),
            detection_batch_count: batch_sink
                .as_ref()
                .map(|_| processed_source_frames)
                .unwrap_or(0),
            scene_snapshot_path: audit.scene_snapshot_output.as_ref().map(|output| {
                match &output.destination {
                    SceneSnapshotOutputDestination::JsonlFile { path } => {
                        path.display().to_string()
                    }
                }
            }),
            scene_snapshot_count,
            detailed_trace_path: options
                .details_jsonl_path
                .as_ref()
                .map(|path| path.display().to_string()),
            detailed_trace_written: trace_written,
            detailed_trace_dropped_by_limit: trace_limited,
            phase_cue_geometry_trace_path: options
                .phase_cue_geometry_jsonl_path
                .as_ref()
                .map(|path| path.display().to_string()),
            phase_cue_geometry_trace_written: cue_geometry_trace_written,
            phase_cue_geometry_trace_dropped_by_limit: cue_geometry_trace_limited,
            artifact_writes_included_in_measurement: audit.detection_batch_output.is_some()
                || audit.scene_snapshot_output.is_some()
                || options.details_jsonl_path.is_some()
                || options.phase_cue_geometry_jsonl_path.is_some(),
            offline_decode_excluded_from_measurement: true,
        },
        effective_tracking_configuration: audit.scene_snapshot_output.as_ref().map(|output| {
            serde_json::json!({
                "ball_class_id": output.tracking.ball_class_id(),
                "cue_class_id": output.tracking.cue_class_id(),
                "camera_profile_id": output.tracking.camera_profile_id(),
                "association_gate_px": output.tracking.association_gate_px(),
                "association_prediction_horizon_ms": output.tracking.association_prediction_horizon().as_secs_f64() * 1000.0,
                "lost_after_ms": output.tracking.lost_after().as_secs_f64() * 1000.0,
                "maximum_track_age_ms": output.tracking.maximum_track_age().as_secs_f64() * 1000.0,
                "maximum_active_tracks": output.tracking.maximum_active_tracks(),
                "recent_observation_capacity": output.tracking.recent_observation_capacity(),
                "configuration_revision": output.tracking.configuration_revision(),
                "rectification_revision": output.tracking.rectification_revision(),
            })
        }),
        calibration: audit.calibration.as_ref().map(|options| {
            summarize_calibration_counts(
                options,
                audit.source_image_space,
                processed_source_frames,
                calibrated_frame_count,
                calibrated_detection_count,
            )
        }),
        batch_engine_instance_id: audit
            .detection_batch_output
            .as_ref()
            .map(|output| output.engine_instance_id.clone()),
        source_session_id: audit
            .detection_batch_output
            .as_ref()
            .map(|output| output.source_session_id.clone()),
        temporal_accuracy: None,
        unsupported_measurements: {
            let mut unsupported = vec![
            "camera_exposure_to_decode_latency",
            "browser_render_and_physical_display_latency",
            "process_cpu_utilization_and_memory_trend",
            "accelerator_per_node_assignment_for_native_openvino",
            ];
            if audit.calibration.is_some()
                && options.mode == VideoReplayMode::Capacity
            {
                unsupported.push("source_to_calibration_latency_for_unpaced_capacity_replay");
            }
            unsupported
        },
    };
    Ok(report)
}

fn validate_scene_classes(
    scene_options: Option<&SceneSnapshotOutputOptions>,
    class_names: &[String],
) -> Result<()> {
    if let Some(scene_options) = scene_options {
        ensure!(
            (scene_options.tracking.ball_class_id() as usize) < class_names.len(),
            "scene tracking ball class ID is absent from the model manifest"
        );
        if let Some(cue_class_id) = scene_options.tracking.cue_class_id() {
            ensure!(
                (cue_class_id as usize) < class_names.len(),
                "scene tracking cue class ID is absent from the model manifest"
            );
        }
    }
    Ok(())
}

fn executable_version(executable: &Path, timeout: Duration) -> Result<String> {
    let mut command = Command::new(executable);
    command.arg("-version");
    let output = run_bounded_command(command, timeout, 8 * 1024, "version query")?;
    let first_line = String::from_utf8_lossy(&output)
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_owned();
    ensure!(
        !first_line.is_empty(),
        "{} returned no version string",
        executable.display()
    );
    Ok(first_line)
}

fn probe_global_video_stream_index(
    ffprobe: &Path,
    video_path: &Path,
    video_stream_ordinal: u32,
    timeout: Duration,
) -> Result<u32> {
    let mut command = Command::new(ffprobe);
    command
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg(format!("v:{video_stream_ordinal}"))
        .arg("-show_entries")
        .arg("stream=index")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(video_path);
    let output = run_bounded_command(command, timeout, 1024, "selected video stream query")?;
    let selected_index = std::str::from_utf8(&output)
        .context("ffprobe stream index was not UTF-8")?
        .trim()
        .parse::<u32>()
        .context("ffprobe did not return one selected global stream index")?;
    Ok(selected_index)
}

fn probe_video_timeline(
    ffprobe: &Path,
    video_path: &Path,
    video_stream_ordinal: u32,
    maximum_output_bytes: usize,
    timeout: Duration,
) -> Result<Vec<u8>> {
    let mut command = Command::new(ffprobe);
    command
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg(format!("v:{video_stream_ordinal}"))
        .arg("-show_streams")
        .arg("-show_frames")
        .arg("-show_entries")
        .arg("stream=index,codec_type,width,height:stream_tags=rotate:stream_side_data=rotation,displaymatrix:frame=width,height,best_effort_timestamp_time:frame_side_data=rotation,displaymatrix")
        .arg("-of")
        .arg("json")
        .arg(video_path);
    run_bounded_command(
        command,
        timeout,
        maximum_output_bytes,
        "video timeline query",
    )
}

fn run_bounded_command(
    mut command: Command,
    timeout: Duration,
    maximum_stdout_bytes: usize,
    label: &'static str,
) -> Result<Vec<u8>> {
    ensure!(!timeout.is_zero(), "{label} timeout must be positive");
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("could not start {label}"))?;
    let stdout = child
        .stdout
        .take()
        .context("child stdout pipe was unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("child stderr pipe was unavailable")?;
    let overflow = Arc::new(AtomicBool::new(false));
    let stdout_overflow = Arc::clone(&overflow);
    let stdout_reader = thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(maximum_stdout_bytes.min(8 * 1024 * 1024));
        let mut chunk = [0u8; 64 * 1024];
        let mut reader = stdout;
        loop {
            let count = reader.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            let retained = maximum_stdout_bytes.saturating_sub(output.len()).min(count);
            output.extend_from_slice(&chunk[..retained]);
            if retained != count {
                stdout_overflow.store(true, Ordering::Relaxed);
            }
        }
        Ok(output)
    });
    let stderr_reader = thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(MAXIMUM_VIDEO_CHILD_STDERR_BYTES);
        let mut chunk = [0u8; 4096];
        let mut reader = stderr;
        loop {
            let count = reader.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            let retained = MAXIMUM_VIDEO_CHILD_STDERR_BYTES
                .saturating_sub(output.len())
                .min(count);
            output.extend_from_slice(&chunk[..retained]);
        }
        Ok(output)
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(error).with_context(|| format!("could not check {label} status"));
            }
        }
        if overflow.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!("{label} output exceeds configured size bound ({maximum_stdout_bytes} bytes)");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!("{label} exceeded configured timeout {timeout:?}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("{label} stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("{label} stderr reader panicked"))??;
    if overflow.load(Ordering::Relaxed) {
        bail!("{label} output exceeds configured size bound ({maximum_stdout_bytes} bytes)");
    }
    if !status.success() {
        let detail = String::from_utf8_lossy(&stderr);
        let detail = detail.trim();
        if detail.is_empty() {
            bail!("{label} returned a non-zero exit status");
        }
        bail!("{label} failed: {detail}");
    }
    Ok(stdout)
}

fn decode_video_to_spool(
    ffmpeg: &Path,
    video_path: &Path,
    global_stream_index: u32,
    expected_decoded_bytes: u64,
    maximum_decoded_bytes: u64,
    timeout: Duration,
) -> Result<(File, f64, u64)> {
    ensure!(
        expected_decoded_bytes <= maximum_decoded_bytes,
        "decoded video exceeds configured raw RGB spool size"
    );
    let raw_spool = tempfile::tempfile().context("could not create temporary RGB video spool")?;
    let mut command = Command::new(ffmpeg);
    command
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-xerror")
        .arg("-nostdin")
        .arg("-noautorotate")
        .arg("-i")
        .arg(video_path)
        .arg("-map")
        .arg(format!("0:{global_stream_index}"))
        .arg("-fps_mode")
        .arg("passthrough")
        .arg("-pix_fmt")
        .arg("rgb24")
        .arg("-f")
        .arg("rawvideo")
        .arg("-");
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("could not start ffmpeg executable {}", ffmpeg.display()))?;
    let stdout = child
        .stdout
        .take()
        .context("ffmpeg stdout pipe was unavailable")?;
    let stderr = child
        .stderr
        .take()
        .context("ffmpeg stderr pipe was unavailable")?;
    let overflow = Arc::new(AtomicBool::new(false));
    let stdout_overflow = Arc::clone(&overflow);
    let stdout_reader = thread::spawn(move || -> std::io::Result<(File, u64)> {
        let mut spool = raw_spool;
        let mut reader = stdout;
        let mut total = 0u64;
        let mut chunk = [0u8; 256 * 1024];
        loop {
            let count = reader.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            let next_total = total.saturating_add(count as u64);
            if next_total > expected_decoded_bytes || next_total > maximum_decoded_bytes {
                stdout_overflow.store(true, Ordering::Relaxed);
            }
            let retained = expected_decoded_bytes
                .saturating_sub(total)
                .min(count as u64) as usize;
            if retained > 0 {
                spool.write_all(&chunk[..retained])?;
            }
            total = next_total;
        }
        spool.flush()?;
        Ok((spool, total))
    });
    let stderr_reader = thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut output = Vec::with_capacity(MAXIMUM_VIDEO_CHILD_STDERR_BYTES);
        let mut reader = stderr;
        let mut chunk = [0u8; 4096];
        loop {
            let count = reader.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            let retained = MAXIMUM_VIDEO_CHILD_STDERR_BYTES
                .saturating_sub(output.len())
                .min(count);
            output.extend_from_slice(&chunk[..retained]);
        }
        Ok(output)
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(error).context("could not check ffmpeg status");
            }
        }
        if overflow.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!("ffmpeg raw RGB output exceeds the expected or configured byte limit");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!("ffmpeg exceeded configured timeout {timeout:?}");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let (spool, total_bytes) = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("ffmpeg stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("ffmpeg stderr reader panicked"))??;
    let elapsed_ms = elapsed_ms(started);
    if overflow.load(Ordering::Relaxed) {
        bail!("ffmpeg raw RGB output exceeds the expected or configured byte limit");
    }
    if !status.success() {
        let detail = String::from_utf8_lossy(&stderr);
        let detail = detail.trim();
        if detail.is_empty() {
            bail!("ffmpeg raw video decode returned a non-zero exit status");
        }
        bail!("ffmpeg raw video decode failed: {detail}");
    }
    ensure!(
        total_bytes == expected_decoded_bytes,
        "ffmpeg decoded {total_bytes} raw RGB bytes; expected {expected_decoded_bytes} bytes"
    );
    Ok((spool, elapsed_ms, total_bytes))
}

fn read_spooled_rgb_frame(
    spool: &mut File,
    frame_index: usize,
    frame_bytes: usize,
    width: u32,
    height: u32,
) -> Result<RgbImage> {
    let offset = u64::try_from(frame_index)
        .ok()
        .and_then(|index| {
            u64::try_from(frame_bytes)
                .ok()
                .and_then(|bytes| index.checked_mul(bytes))
        })
        .context("video frame spool offset overflowed")?;
    spool
        .seek(SeekFrom::Start(offset))
        .context("could not seek to requested RGB frame")?;
    let mut pixels = vec![0u8; frame_bytes];
    spool
        .read_exact(&mut pixels)
        .with_context(|| format!("RGB frame {frame_index} is truncated in the decoded spool"))?;
    RgbImage::from_raw(width, height, pixels)
        .context("RGB frame dimensions do not match the decoded pixel buffer")
}

fn sha256_file_bounded(path: &Path, maximum_bytes: u64) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("could not hash video {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .context("could not read video while hashing")?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .context("video byte count overflowed")?;
        ensure!(
            total <= maximum_bytes,
            "video exceeds configured hash size limit ({maximum_bytes} bytes)"
        );
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn timeline_media_duration(timeline: &VideoTimeline) -> Option<f64> {
    if timeline.frame_count() < 2 {
        return None;
    }
    let mut intervals = timeline
        .presentation_times_seconds
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    intervals.sort_by(f64::total_cmp);
    let midpoint = intervals.len() / 2;
    let final_interval = if intervals.len() % 2 == 0 {
        (intervals[midpoint - 1] + intervals[midpoint]) / 2.0
    } else {
        intervals[midpoint]
    };
    timeline
        .presentation_times_seconds
        .last()
        .copied()
        .map(|last_pts| last_pts + final_interval)
        .filter(|duration| duration.is_finite() && *duration > 0.0)
}

/// Return the newest source frame whose PTS has arrived, without moving back
/// before the next unprocessed frame. This models a capacity-one latest-frame
/// mailbox after inference has yielded to the replay scheduler.
fn latest_due_source_index(
    presentation_times_seconds: &[f64],
    next_source_index: usize,
    elapsed_source_seconds: f64,
) -> usize {
    debug_assert!(next_source_index < presentation_times_seconds.len());
    presentation_times_seconds
        .partition_point(|presentation_time| *presentation_time <= elapsed_source_seconds)
        .saturating_sub(1)
        .max(next_source_index)
}

fn checked_duration_from_seconds(seconds: f64) -> Result<Duration> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0 && seconds <= u64::MAX as f64,
        "video duration is outside the monotonic clock range"
    );
    let whole_seconds = seconds.floor() as u64;
    let fractional_nanoseconds = ((seconds - whole_seconds as f64) * 1_000_000_000.0).round();
    if fractional_nanoseconds >= 1_000_000_000.0 {
        return whole_seconds
            .checked_add(1)
            .map(|whole| Duration::new(whole, 0))
            .context("video duration exceeds the monotonic clock range");
    }
    Ok(Duration::new(whole_seconds, fractional_nanoseconds as u32))
}

fn video_frame_deadline(start: Instant, presentation_time_seconds: f64) -> Result<Instant> {
    let offset = checked_duration_from_seconds(presentation_time_seconds)?;
    start
        .checked_add(offset)
        .context("video presentation deadline exceeds monotonic clock range")
}

fn completed_after_source_deadline(
    source_start: Instant,
    next_source_presentation_time_seconds: f64,
    completed_at: Instant,
) -> Result<bool> {
    Ok(completed_at > video_frame_deadline(source_start, next_source_presentation_time_seconds)?)
}

fn wait_until(deadline: Instant) {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if !remaining.is_zero() {
        thread::sleep(remaining);
    }
}

fn create_new_buffered_file(path: &Path, label: &str) -> Result<BufWriter<File>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {label} directory {}", parent.display()))?;
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "could not create {label} {} without overwriting",
                path.display()
            )
        })?;
    Ok(BufWriter::new(file))
}

#[derive(Serialize)]
struct VideoFrameDetail<'a> {
    source_frame_index: usize,
    source_presentation_time_seconds: f64,
    source_frame_id: String,
    source_to_inference_start_ms: Option<f64>,
    source_to_inference_complete_ms: Option<f64>,
    timings: &'a FrameTimings,
    detections: &'a [Detection],
}

#[derive(Serialize)]
struct VideoCueGeometryDetail {
    source_video_sha256: String,
    source_frame_index: usize,
    source_presentation_time_seconds: f64,
    cue_present: bool,
    shaft_start_px: Option<[f64; 2]>,
    shaft_end_px: Option<[f64; 2]>,
    axis_angle_degrees: Option<f64>,
    confidence: Option<f64>,
    signed_direction: Option<&'static str>,
}

fn write_cue_geometry_trace<W: Write>(
    writer: &mut W,
    source_video_sha256: &str,
    source_frame_index: usize,
    source_pts_seconds: f64,
    cue: Option<CueGeometry>,
) -> Result<()> {
    let detail = VideoCueGeometryDetail {
        source_video_sha256: source_video_sha256.to_owned(),
        source_frame_index,
        source_presentation_time_seconds: source_pts_seconds,
        cue_present: cue.is_some(),
        shaft_start_px: cue.map(|geometry| geometry.shaft_start_px),
        shaft_end_px: cue.map(|geometry| geometry.shaft_end_px),
        axis_angle_degrees: cue.map(|geometry| {
            geometry.axis_unit[1]
                .atan2(geometry.axis_unit[0])
                .to_degrees()
        }),
        confidence: cue.map(|geometry| geometry.confidence),
        signed_direction: cue.and_then(|geometry| match geometry.signed_direction {
            Some(SignedCueDirection::TowardStart) => Some("toward_start"),
            Some(SignedCueDirection::TowardEnd) => Some("toward_end"),
            None => None,
        }),
    };
    serde_json::to_writer(&mut *writer, &detail)
        .context("could not encode phase cue geometry trace")?;
    writer
        .write_all(b"\n")
        .context("could not delimit phase cue geometry trace")?;
    Ok(())
}

fn write_video_detail<W: Write>(
    writer: &mut W,
    source_frame_index: usize,
    source_pts_seconds: f64,
    result: &FrameResult,
) -> Result<()> {
    let detail = VideoFrameDetail {
        source_frame_index,
        source_presentation_time_seconds: source_pts_seconds,
        source_frame_id: format!("frame-{:06}", source_frame_index + 1),
        source_to_inference_start_ms: result.source_to_inference_start_ms,
        source_to_inference_complete_ms: result.source_to_inference_complete_ms,
        timings: &result.timings,
        detections: &result.detections,
    };
    serde_json::to_writer(&mut *writer, &detail).context("could not encode video detail trace")?;
    writer
        .write_all(b"\n")
        .context("could not delimit video detail trace")?;
    Ok(())
}

fn replay_cue_proposals(
    detections: &[Detection],
    width_px: u32,
    height_px: u32,
    maximum_proposals: usize,
) -> Vec<CueProposal> {
    detections
        .iter()
        .filter(|detection| detection.class_name.eq_ignore_ascii_case("cue"))
        .take(maximum_proposals)
        .filter_map(|detection| {
            let [left, top, right, bottom] = detection.bbox_xyxy_px;
            let proposal = CueProposal {
                left: left.floor().clamp(0.0, width_px as f32) as u32,
                top: top.floor().clamp(0.0, height_px as f32) as u32,
                right: right.ceil().clamp(0.0, width_px as f32) as u32,
                bottom: bottom.ceil().clamp(0.0, height_px as f32) as u32,
            };
            (proposal.right > proposal.left && proposal.bottom > proposal.top).then_some(proposal)
        })
        .collect()
}

fn replay_schedule_value(config: &SchedulerConfig) -> Value {
    fn cadence(value: crate::phase_scheduler::PhaseCadence) -> Value {
        serde_json::json!({
            "motion_hz": value.motion_hz,
            "ball_hz": value.ball_hz,
            "cue_hz": value.cue_hz,
        })
    }

    serde_json::json!({
        "recovering": cadence(config.recovering),
        "pre_shot": cadence(config.pre_shot),
        "aiming": cadence(config.aiming),
        "shot": cadence(config.shot),
        "quiet_dwell_ms": config.quiet_dwell.as_secs_f64() * 1000.0,
        "layout_consensus_count": config.layout_consensus_count,
        "layout_position_tolerance_px": config.layout_position_tolerance_px,
        "motion_evidence_expiry_ms": config.motion_evidence_expiry.as_secs_f64() * 1000.0,
        "cue_evidence_expiry_ms": config.cue_evidence_expiry.as_secs_f64() * 1000.0,
        "layout_evidence_expiry_ms": config.layout_evidence_expiry.as_secs_f64() * 1000.0,
    })
}

fn mean_or_zero(total: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[allow(clippy::too_many_arguments)]
fn process_frame(
    context: &mut FrameProcessContext<'_>,
    frame: FrameToProcess,
    batch_sequence: usize,
) -> Result<(FrameResult, f64, Instant)> {
    let inference_start = Instant::now();
    let inference_output = context
        .inference_engine
        .infer_prepared(&frame.timed_input.prepared)
        .with_context(|| format!("failed to infer {}", frame.path.display()))?;
    let inference_ms = inference_output.timings.inference_ms;
    let decode_and_nms_ms = inference_output.timings.decode_and_nms_ms;
    let detections = inference_output.detections;
    let inference_completed = Instant::now();
    let end_to_end_ms = frame.timed_input.decode_ms
        + frame.timed_input.preprocess_ms
        + inference_ms
        + decode_and_nms_ms;
    // Only a paced source has a meaningful presentation/arrival-to-inference
    // latency. Capacity replay still supplies a local monotonic arrival to the
    // tracker, but its timestamps are not a freshness measurement.
    let source_to_inference_start_ms =
        frame
            .scheduled_at
            .zip(frame.source_arrived_at)
            .map(|(_, source_arrived_at)| {
                inference_start
                    .saturating_duration_since(source_arrived_at)
                    .as_secs_f64()
                    * 1000.0
            });
    let source_to_inference_complete_ms =
        frame
            .scheduled_at
            .zip(frame.source_arrived_at)
            .map(|(_, source_arrived_at)| {
                inference_completed
                    .saturating_duration_since(source_arrived_at)
                    .as_secs_f64()
                    * 1000.0
            });
    let mut source_to_calibration_complete_ms = None;

    let mut batch_export_ms = 0.0;
    let mut calibration_ms = None;
    let mut ball_refinement_ms = None;
    let mut calibration_report = None;
    let mut scene_tracking_ms = None;
    let mut scene_encode_enqueue_ms = None;
    if let (Some(sink), Some(output_options)) = (
        context.batch_sink.as_mut(),
        context.options.detection_batch_output.as_ref(),
    ) {
        let export_start = Instant::now();
        let batch = build_detection_batch(
            frame.source_index,
            batch_sequence,
            frame.timed_input.image.width(),
            frame.timed_input.image.height(),
            &detections,
            context.class_names,
            context.onnx_sha256,
            context.source_checkpoint_sha256,
            context.options,
            output_options,
            context.batch_clock_origin,
            inference_start,
            inference_completed,
            frame.source_arrived_at,
        )?;
        write_detection_batch(sink, &batch)?;
        batch_export_ms = elapsed_ms(export_start);

        let replay_tracking_context = context.scene_output.as_ref().map(|scene| {
            (
                scene.tracking.ball_class_id(),
                scene.tracking.camera_profile_id().to_owned(),
                scene.tracking.configuration_revision(),
            )
        });
        let mut replay_refinements = None;
        if let (Some(profile), Some((ball_class_id, camera_profile_id, configuration_revision))) =
            (context.localization_profile, replay_tracking_context)
        {
            if profile.optics.is_some() {
                let refinement_started = Instant::now();
                replay_refinements = match refine_replay_ball_batch(
                    &batch,
                    frame.timed_input.image.as_raw(),
                    profile,
                    ball_class_id,
                    &camera_profile_id,
                    configuration_revision,
                ) {
                    Ok(refinements) => Some(refinements),
                    Err(error) => {
                        eprintln!(
                            "replay ball refinement unavailable for {}: {error:#}",
                            batch.source.frame_id
                        );
                        None
                    }
                };
                ball_refinement_ms = Some(elapsed_ms(refinement_started));
            }
        }

        if let Some(calibration_options) = &context.options.calibration {
            let calibration_start = Instant::now();
            let applied_context = AppliedCalibrationContext {
                calibration_revision: calibration_options.applied_calibration_revision,
                camera_profile_id: &calibration_options.applied_camera_profile_id,
                rectification_revision: calibration_options.applied_rectification_revision,
            };
            let geometry_result = calibrate_detection_batch(
                &batch,
                &calibration_options.calibration,
                applied_context,
            );
            calibration_report = Some(calibration_report_from_geometry_result(&geometry_result)?);
            calibration_ms = Some(elapsed_ms(calibration_start));
            source_to_calibration_complete_ms = paced_source_to_calibration_latency_ms(
                frame.scheduled_at,
                frame.source_arrived_at,
                Instant::now(),
            );
        }

        if let Some(scene_runtime) = context.scene_output.as_mut() {
            let published_at = scene_publication_timestamp(
                context.batch_clock_origin,
                &output_options.engine_instance_id,
            )?;
            let tracking_start = Instant::now();
            // Replay and live tracking use the same frame-bound selection and
            // table-mapping helper. Missing or mismatched fits retain only the
            // raw midpoint for visibility.
            let selected_measurements = resolve_ball_measurements_for_batch(
                &batch,
                crate::ball_localization::BallMeasurementResolutionContext {
                    ball_class_id: scene_runtime.tracking.ball_class_id(),
                    camera_profile_id: scene_runtime.tracking.camera_profile_id(),
                    configuration_revision: scene_runtime.tracking.configuration_revision(),
                    rectification_revision: scene_runtime.tracking.rectification_revision(),
                    profile: context.localization_profile,
                    refinements: replay_refinements.as_ref(),
                    expected_identity: None,
                    require_identity: false,
                },
            );
            let tracked_scene = scene_runtime
                .tracker
                .process_with_resolved_ball_measurements(
                    &batch,
                    published_at,
                    &selected_measurements,
                )
                .context("failed to produce tracked scene snapshot")?;
            let mut snapshot = tracked_scene.snapshot;
            scene_tracking_ms = Some(elapsed_ms(tracking_start));
            snapshot.published_at = scene_publication_timestamp(
                context.batch_clock_origin,
                &output_options.engine_instance_id,
            )?;
            let export_start = Instant::now();
            write_scene_snapshot(&mut scene_runtime.sink, &snapshot)?;
            scene_encode_enqueue_ms = Some(elapsed_ms(export_start));
            scene_runtime.snapshot_count += 1;
            scene_runtime.total_tracking_ms += scene_tracking_ms.unwrap_or_default();
        }
    }

    Ok((
        FrameResult {
            timings: FrameTimings {
                image_decode_ms: frame.timed_input.decode_ms,
                preprocess_ms: frame.timed_input.preprocess_ms,
                inference_ms,
                decode_and_nms_ms,
                end_to_end_ms,
                ball_refinement_ms,
                calibration_ms,
                scene_tracking_ms,
                scene_encode_enqueue_ms,
            },
            source_to_inference_start_ms,
            source_to_inference_complete_ms,
            source_to_calibration_complete_ms,
            detections,
            calibration: calibration_report,
        },
        batch_export_ms,
        inference_completed,
    ))
}

fn refine_replay_ball_batch(
    batch: &DetectionBatchV1,
    rgb: &[u8],
    profile: &LoadedBallLocalizationProfile,
    ball_class_id: u32,
    camera_profile_id: &str,
    configuration_revision: u64,
) -> Result<BallRefinementBatch> {
    let optics = profile
        .optics
        .as_ref()
        .context("camera optics profile is unavailable")?;
    ensure!(
        (
            optics.intrinsics.source_width_px,
            optics.intrinsics.source_height_px
        ) == (batch.source.width_px, batch.source.height_px),
        "source frame dimensions differ from the loaded camera optics profile"
    );
    let proposals = batch
        .detections
        .iter()
        .filter(|detection| detection.class_id == ball_class_id)
        .map(|detection| BallProposal {
            detection_index: detection.detection_index,
            bbox: [
                detection.bbox_image_px.left,
                detection.bbox_image_px.top,
                detection.bbox_image_px.right,
                detection.bbox_image_px.bottom,
            ],
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let observations = refine_ball_batch(
        rgb,
        &optics.intrinsics,
        &proposals,
        &profile.settings.parameters,
    )
    .map_err(|error| anyhow::anyhow!("native-resolution refinement failed: {error}"))?;
    Ok(BallRefinementBatch {
        source_width_px: batch.source.width_px,
        source_height_px: batch.source.height_px,
        camera_profile_id: camera_profile_id.to_owned(),
        image_space: batch.image_space,
        configuration_revision,
        optics_profile_id: optics.optics_profile_id.clone(),
        optics_revision: optics.optics_revision,
        refiner_revision: optics.refiner_revision,
        table_revision: profile.settings.table_revision,
        refinement_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        identity: None,
        observations,
    })
}

#[cfg(test)]
fn calibration_report_for_batch(
    batch: &DetectionBatchV1,
    options: &CalibrationAuditOptions,
) -> Result<FrameCalibrationReport> {
    let applied_context = AppliedCalibrationContext {
        calibration_revision: options.applied_calibration_revision,
        camera_profile_id: &options.applied_camera_profile_id,
        rectification_revision: options.applied_rectification_revision,
    };
    let result = calibrate_detection_batch(batch, &options.calibration, applied_context);
    calibration_report_from_geometry_result(&result)
}

fn calibration_report_from_geometry_result(
    result: &std::result::Result<crate::geometry::CalibratedDetectionBatch, GeometryError>,
) -> Result<FrameCalibrationReport> {
    match result {
        Ok(calibrated) => Ok(FrameCalibrationReport {
            status: CalibrationStatus::Valid,
            reason: None,
            projected_box_midpoint_count: Some(calibrated.detections.len()),
        }),
        Err(error @ GeometryError::IncompatibleFrameCalibration(_))
        | Err(error @ GeometryError::DetectionProjection { .. }) => Ok(FrameCalibrationReport {
            status: CalibrationStatus::Unavailable,
            reason: Some(error.to_string()),
            projected_box_midpoint_count: None,
        }),
        Err(error) => {
            bail!("audit-created detection batch violated geometry input contract: {error}")
        }
    }
}

fn percentile_summary(mut samples_ms: Vec<f64>) -> LatencyPercentiles {
    samples_ms.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        if samples_ms.is_empty() {
            None
        } else {
            let rank = (fraction * samples_ms.len() as f64).ceil() as usize;
            Some(samples_ms[rank.saturating_sub(1).min(samples_ms.len() - 1)])
        }
    };
    LatencyPercentiles {
        sample_count: samples_ms.len(),
        p50_ms: percentile(0.50),
        p95_ms: percentile(0.95),
        p99_ms: percentile(0.99),
    }
}

fn validate_video_audit_options(options: &AuditOptions) -> Result<()> {
    ensure!(
        options.onnx_path.is_file(),
        "ONNX model does not exist: {}",
        options.onnx_path.display()
    );
    ensure!(
        options.source_checkpoint_path.is_file(),
        "source checkpoint does not exist: {}",
        options.source_checkpoint_path.display()
    );
    validate_inference_parameters(
        options.image_size,
        options.confidence,
        options.iou,
        options.max_detections,
        options.intra_op_threads,
    )?;
    if let Some(batch_output) = &options.detection_batch_output {
        validate_batch_identifier(&batch_output.engine_instance_id, "engine instance ID")?;
        validate_batch_identifier(&batch_output.source_session_id, "source session ID")?;
        ensure!(
            !batch_output.jsonl_path.exists(),
            "detection batch output already exists: {}",
            batch_output.jsonl_path.display()
        );
    }
    if let Some(calibration) = &options.calibration {
        ensure!(
            options.detection_batch_output.is_some(),
            "calibration audit requires detection-batch output so raw results are reproducible"
        );
        ensure!(
            calibration.bundle_path.is_file(),
            "calibration bundle does not exist: {}",
            calibration.bundle_path.display()
        );
        validate_batch_identifier(
            &calibration.applied_camera_profile_id,
            "applied camera profile ID",
        )?;
        ensure!(
            calibration.applied_calibration_revision <= MAX_SAFE_JSON_INTEGER
                && calibration.applied_rectification_revision <= MAX_SAFE_JSON_INTEGER,
            "applied calibration revisions must be safe JSON integers"
        );
        ensure!(
            calibration.bundle_sha256.len() == 64
                && calibration
                    .bundle_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "calibration bundle SHA-256 must be 64 hexadecimal characters"
        );
    }
    if let Some(scene_output) = &options.scene_snapshot_output {
        ensure!(
            options.detection_batch_output.is_some(),
            "scene snapshot output requires detection-batch output to preserve raw observations"
        );
        let SceneSnapshotOutputDestination::JsonlFile { path } = &scene_output.destination;
        ensure!(
            !path.exists(),
            "scene snapshot output already exists: {}",
            path.display()
        );
        if let Some(BatchOutputOptions {
            jsonl_path: batch_path,
            ..
        }) = &options.detection_batch_output
        {
            ensure!(
                batch_path != path,
                "detection batch and scene snapshot output must use different paths"
            );
        }
        if let Some(calibration) = &options.calibration {
            validate_scene_calibration_context(&scene_output.tracking, calibration)?;
        }
    }
    Ok(())
}

fn validate_frame_inference_options(options: &FrameInferenceOptions) -> Result<()> {
    let model_path = options
        .onnx_path
        .as_deref()
        .context("onnx_path is required for the selected provider")?;
    ensure!(
        model_path.is_file(),
        "ONNX model does not exist: {}",
        model_path.display()
    );
    ensure!(
        options.source_checkpoint_path.is_file(),
        "source checkpoint does not exist: {}",
        options.source_checkpoint_path.display()
    );
    validate_inference_parameters(
        options.image_size,
        options.confidence,
        options.iou,
        options.max_detections,
        options.intra_op_threads,
    )
}

fn validate_inference_parameters(
    image_size: u32,
    confidence: f32,
    iou: f32,
    max_detections: usize,
    intra_op_threads: Option<usize>,
) -> Result<()> {
    ensure!(
        image_size > 0 && image_size.is_multiple_of(32),
        "image size must be a positive multiple of 32"
    );
    ensure!(
        confidence.is_finite() && (0.0..=1.0).contains(&confidence),
        "confidence must be finite and between 0 and 1"
    );
    ensure!(
        iou.is_finite() && (0.0..=1.0).contains(&iou),
        "IoU must be finite and between 0 and 1"
    );
    ensure!(max_detections > 0, "max detections must be positive");
    ensure!(
        max_detections <= u32::MAX as usize,
        "maximum detections exceeds the detection-batch contract limit"
    );
    if let Some(threads) = intra_op_threads {
        ensure!(threads > 0, "intra-op thread count must be positive");
    }
    Ok(())
}

fn validate_scene_calibration_context(
    tracking: &TrackingConfigV1,
    calibration: &CalibrationAuditOptions,
) -> Result<()> {
    ensure!(
        tracking.camera_profile_id() == calibration.applied_camera_profile_id(),
        "scene tracking camera profile must match the applied calibration profile"
    );
    ensure!(
        tracking.rectification_revision() == calibration.applied_rectification_revision(),
        "scene tracking rectification revision must match the applied rectification revision"
    );
    Ok(())
}

fn validate_batch_identifier(value: &str, label: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{label} must not be empty");
    ensure!(
        value.chars().count() <= MAX_IDENTIFIER_CHARACTERS
            && !value.chars().any(char::is_control),
        "{label} must be at most {MAX_IDENTIFIER_CHARACTERS} characters and contain no control characters"
    );
    Ok(())
}

fn open_detection_batch_sink(options: &BatchOutputOptions) -> Result<DetectionBatchSink> {
    if let Some(parent) = options.jsonl_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create detection batch directory {}",
                    parent.display()
                )
            })?;
        }
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&options.jsonl_path)
        .with_context(|| {
            format!(
                "failed to create detection batch output {} without overwriting",
                options.jsonl_path.display()
            )
        })?;
    Ok(BufWriter::new(file))
}

fn open_scene_snapshot_output(
    options: &SceneSnapshotOutputOptions,
) -> Result<SceneSnapshotOutputRuntime> {
    let SceneSnapshotOutputDestination::JsonlFile { path } = &options.destination;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create scene snapshot directory {}",
                    parent.display()
                )
            })?;
        }
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "failed to create scene snapshot output {} without overwriting",
                path.display()
            )
        })?;
    Ok(SceneSnapshotOutputRuntime {
        tracker: SceneTrackerV1::new(options.tracking.clone()),
        tracking: options.tracking.clone(),
        sink: SceneSnapshotSink::Jsonl(BufWriter::new(file)),
        snapshot_count: 0,
        total_tracking_ms: 0.0,
    })
}

#[cfg(test)]
fn write_scene_snapshot_line<W: Write>(writer: &mut W, snapshot: &SceneSnapshotV1) -> Result<()> {
    let json = encode_scene_snapshot_v1(snapshot).context("scene snapshot failed validation")?;
    write_scene_snapshot_jsonl(writer, &json)
}

fn write_scene_snapshot_jsonl<W: Write>(writer: &mut W, json: &[u8]) -> Result<()> {
    writer
        .write_all(json)
        .context("failed to write scene snapshot")?;
    writer
        .write_all(b"\n")
        .context("failed to write scene snapshot delimiter")?;
    writer.flush().context("failed to flush scene snapshot")?;
    Ok(())
}

fn write_scene_snapshot(sink: &mut SceneSnapshotSink, snapshot: &SceneSnapshotV1) -> Result<()> {
    let json = encode_scene_snapshot_v1(snapshot).context("scene snapshot failed validation")?;
    match sink {
        SceneSnapshotSink::Jsonl(writer) => write_scene_snapshot_jsonl(writer, &json),
    }
}

fn write_detection_batch(sink: &mut DetectionBatchSink, batch: &DetectionBatchV1) -> Result<()> {
    write_detection_batch_line(sink, batch)
}

#[allow(clippy::too_many_arguments)]
fn build_detection_batch(
    frame_index: usize,
    batch_sequence: usize,
    source_width_px: u32,
    source_height_px: u32,
    detections: &[Detection],
    class_names: &[String],
    artifact_sha256: &str,
    source_checkpoint_sha256: &str,
    options: &AuditOptions,
    batch_options: &BatchOutputOptions,
    clock_origin: Instant,
    inference_started: Instant,
    inference_completed: Instant,
    source_arrived_at: Option<Instant>,
) -> Result<DetectionBatchV1> {
    let one_based_frame_index = frame_index
        .checked_add(1)
        .context("source frame index overflowed the platform index range")?;
    let source_frame_id = format!("frame-{one_based_frame_index:06}");
    let batch_sequence = u64::try_from(batch_sequence)
        .context("batch sequence exceeds the supported integer range")?;
    let classes = class_names
        .iter()
        .enumerate()
        .map(|(class_id, class_name)| {
            Ok(ModelClass {
                class_id: u32::try_from(class_id)
                    .context("model class ID exceeds the batch contract range")?,
                class_name: class_name.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let batch_detections = detections
        .iter()
        .enumerate()
        .map(|(detection_index, detection)| {
            let [left, top, right, bottom] = detection.bbox_xyxy_px;
            // The contract serializes pixel geometry as f64 and requires the
            // center to be its exact midpoint within a small tolerance. The
            // detector's f32 midpoint can round differently at 4K scale, so
            // derive the published center from the same serialized box.
            let bbox_image_px = BoundingBox {
                left: left as f64,
                top: top as f64,
                right: right as f64,
                bottom: bottom as f64,
            };
            Ok(BatchDetection {
                detection_index: u64::try_from(detection_index)
                    .context("detection index exceeds the batch contract range")?,
                class_id: u32::try_from(detection.class_id)
                    .context("detection class ID exceeds the batch contract range")?,
                class_name: detection.class_name.clone(),
                confidence: detection.confidence as f64,
                bbox_image_px,
                center_image_px: Point {
                    x: (bbox_image_px.left + bbox_image_px.right) / 2.0,
                    y: (bbox_image_px.top + bbox_image_px.bottom) / 2.0,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let timestamp = |instant, stage| {
        batch_timestamp(
            clock_origin,
            instant,
            &batch_options.engine_instance_id,
            stage,
        )
    };

    let batch = DetectionBatchV1 {
        schema_version: 1,
        engine_instance_id: batch_options.engine_instance_id.clone(),
        batch_sequence,
        source: BatchSourceFrame {
            session_id: batch_options.source_session_id.clone(),
            frame_id: source_frame_id,
            // A paced replay's timestamp identifies when a decoded input was
            // made available to the inference worker. It is synthetic frame
            // arrival time, never a camera exposure timestamp.
            timestamp: source_arrived_at
                .map(|at| timestamp(at, TimestampStage::FrameArrival))
                .transpose()?,
            width_px: source_width_px,
            height_px: source_height_px,
        },
        image_space: options.source_image_space,
        model: ModelIdentity {
            artifact_sha256: artifact_sha256.to_owned(),
            source_checkpoint_sha256: Some(source_checkpoint_sha256.to_owned()),
            classes,
        },
        settings: InferenceSettings {
            input_width_px: options.image_size,
            input_height_px: options.image_size,
            confidence_threshold: options.confidence as f64,
            iou_threshold: options.iou as f64,
            max_detections: u32::try_from(options.max_detections)
                .context("maximum detections exceeds the batch contract range")?,
        },
        inference_started_at: timestamp(inference_started, TimestampStage::InferenceStarted)?,
        inference_completed_at: timestamp(inference_completed, TimestampStage::InferenceComplete)?,
        outcome: BatchOutcome::Success,
        reason: None,
        detections: batch_detections,
        // Capture after all batch fields have been assembled and immediately
        // before the caller encodes and writes this publication attempt.
        published_at: timestamp(Instant::now(), TimestampStage::BatchPublication)?,
    };
    Ok(batch)
}

fn batch_timestamp(
    origin: Instant,
    at: Instant,
    domain_id: &str,
    stage: TimestampStage,
) -> Result<BatchTimestamp> {
    let elapsed = at
        .checked_duration_since(origin)
        .context("batch timestamp precedes the replay clock origin")?;
    let ticks = u64::try_from(elapsed.as_nanos())
        .context("process-local nanosecond clock exceeded the batch contract range")?;
    Ok(BatchTimestamp {
        ticks: ticks.to_string(),
        unit: TimeUnit::Nanoseconds,
        clock: ClockKind::Monotonic,
        domain_id: domain_id.to_owned(),
        stage,
    })
}

fn scene_publication_timestamp(
    origin: Instant,
    engine_instance_id: &str,
) -> Result<crate::Timestamp> {
    let elapsed = Instant::now()
        .checked_duration_since(origin)
        .context("scene publication clock preceded the replay clock origin")?;
    let ticks = u64::try_from(elapsed.as_nanos())
        .context("process-local nanosecond clock exceeded the scene contract range")?;
    Ok(crate::Timestamp {
        ticks: ticks.to_string(),
        unit: crate::TimeUnit::Nanoseconds,
        clock: crate::ClockKind::Monotonic,
        domain_id: engine_instance_id.to_owned(),
        stage: crate::TimestampStage::SnapshotPublication,
    })
}

fn write_detection_batch_line<W: Write>(writer: &mut W, batch: &DetectionBatchV1) -> Result<()> {
    let json = encode_detection_batch_v1(batch).context("detection batch failed validation")?;
    writer
        .write_all(&json)
        .context("failed to write detection batch")?;
    writer
        .write_all(b"\n")
        .context("failed to write detection batch delimiter")?;
    writer.flush().context("failed to flush detection batch")?;
    Ok(())
}

fn load_and_verify_manifest(
    model_path: &Path,
    source_checkpoint_path: &Path,
) -> Result<LoadedModelContract> {
    let manifest_path = PathBuf::from(format!("{}.manifest.json", model_path.display()));
    ensure!(
        manifest_path.is_file(),
        "export manifest is required to verify checkpoint lineage and class names: {}",
        manifest_path.display()
    );
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));
    let manifest: ExportManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("invalid export manifest: {}", manifest_path.display()))?;
    ensure!(
        manifest.manifest_schema_version == MODEL_EXPORT_MANIFEST_SCHEMA_VERSION,
        "unsupported ONNX export manifest schema version: {}",
        manifest.manifest_schema_version
    );
    let actual_model_hash = sha256_file(model_path)?;
    ensure!(
        manifest.exported_model.sha256 == actual_model_hash,
        "exported ONNX hash does not match its manifest"
    );
    let actual_source_hash = sha256_file(source_checkpoint_path)?;
    ensure!(
        manifest.source_model.sha256 == actual_source_hash,
        "source checkpoint hash does not match the export manifest"
    );
    ensure!(
        Path::new(&manifest.exported_model.path).file_name() == model_path.file_name(),
        "manifest artifact filename does not match the ONNX file"
    );
    ensure!(
        manifest.model.class_count == EXPECTED_CLASS_COUNT,
        "this prototype only supports the current two-class checkpoint"
    );
    ensure!(
        manifest.model.class_names.len() == EXPECTED_CLASS_COUNT,
        "manifest class map must contain exactly two entries"
    );
    let mut class_entries = manifest.model.class_names;
    class_entries.sort_by_key(|entry| entry.id);
    ensure!(
        class_entries
            .iter()
            .enumerate()
            .all(|(expected, entry)| entry.id == expected),
        "manifest class IDs must be contiguous from zero"
    );
    let class_names: Vec<String> = class_entries.into_iter().map(|entry| entry.name).collect();
    ensure!(
        class_names.iter().all(|name| !name.trim().is_empty()),
        "manifest class names cannot be empty"
    );

    Ok(LoadedModelContract {
        manifest_path,
        manifest_sha256,
        artifact_sha256: actual_model_hash,
        source_checkpoint_sha256: manifest.source_model.sha256,
        class_names,
        output_coordinate_scale: 1.0,
    })
}

fn create_session(
    environment: &ort::environment::Environment,
    options: &AuditOptions,
    profiling_path: Option<&Path>,
) -> Result<InferenceSession> {
    create_session_for_model(
        environment,
        &options.onnx_path,
        options.provider,
        options.intra_op_threads,
        profiling_path,
    )
}

fn create_session_for_model(
    environment: &Environment,
    onnx_path: &Path,
    provider: ProviderSelection,
    intra_op_threads: Option<usize>,
    profiling_path: Option<&Path>,
) -> Result<InferenceSession> {
    if provider == ProviderSelection::Openvino {
        ensure!(
            profiling_path.is_none(),
            "ONNX Runtime profiling cannot be requested for the native OpenVINO backend"
        );
        return OpenVinoSession::new(onnx_path).map(InferenceSession::Openvino);
    }

    let builder = SessionBuilder::new(environment)?;
    let mut builder = match provider {
        ProviderSelection::Cpu => {
            // `ort` defaults to automatic device selection. On macOS that can
            // register CoreML even when CPU is the only manually requested EP,
            // which makes a purported CPU baseline a mixed-provider run.
            builder
                .with_auto_device(AutoDevicePolicy::PreferCPU)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
                .with_execution_providers([CPU::default().build()])
                .map_err(|error| anyhow::anyhow!(error.to_string()))?
        }
        ProviderSelection::Auto => builder
            .with_auto_device(AutoDevicePolicy::MaxEfficiency)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?,
        ProviderSelection::Openvino => unreachable!("native OpenVINO returned above"),
    };
    if let Some(threads) = intra_op_threads {
        builder = builder
            .with_intra_threads(threads)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    if let Some(path) = profiling_path {
        builder = builder
            .with_profiling(path)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    builder
        .commit_from_file(onnx_path)
        .with_context(|| format!("failed to open ONNX model {}", onnx_path.display()))
        .map(InferenceSession::Ort)
}

/// A single-thread-owned inference backend. OpenVINO is called directly through
/// its C API bindings; it is not registered as an ONNX Runtime execution provider.
enum InferenceSession {
    Ort(Session),
    Openvino(OpenVinoSession),
}

/// Reuses an OpenVINO infer request and its input tensor across frames. The
/// binding's safe tensor API copies prepared input into owned storage, avoiding
/// unsound borrowed buffers across the native boundary.
struct OpenVinoSession {
    request: InferRequest,
    input_tensor: Tensor,
    input_name: String,
    output_name: String,
    input_shape: Vec<i64>,
    output_shape: Vec<i64>,
    execution_devices: Vec<String>,
    _compiled_model: openvino::CompiledModel,
    // Keep Core alive until after the compiled model and infer request are dropped.
    _core: Core,
}

#[cfg(target_os = "macos")]
const OPENVINO_LIBRARY_PATH_VARIABLE: Option<&str> = Some("DYLD_LIBRARY_PATH");
#[cfg(target_os = "linux")]
const OPENVINO_LIBRARY_PATH_VARIABLE: Option<&str> = Some("LD_LIBRARY_PATH");
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
const OPENVINO_LIBRARY_PATH_VARIABLE: Option<&str> = None;

/// Load OpenVINO through its supported runtime discovery first, then handle
/// version-suffixed libraries used by the official macOS PyPI distribution.
/// The fallback is limited to explicitly configured platform library paths so
/// a missing runtime cannot silently select another inference backend.
fn load_openvino_c_library() -> Result<()> {
    if let Some(path) = openvino_sys::library::find() {
        return openvino_sys::library::load_from(&path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load OpenVINO C API library {}", path.display()));
    }

    let Some(path_variable) = OPENVINO_LIBRARY_PATH_VARIABLE else {
        bail!("version-suffixed OpenVINO runtime discovery is supported only on macOS and Linux");
    };
    let library_directories = std::env::var_os(path_variable)
        .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default();
    let library = library_directories
        .into_iter()
        .find_map(|directory| find_versioned_openvino_c_library(&directory));
    let Some(path) = library else {
        bail!(
            "could not find the OpenVINO C API library; install a supported OpenVINO Runtime and add its library directory to {path_variable}"
        );
    };

    openvino_sys::library::load_from(&path)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to load OpenVINO C API library {}", path.display()))
}

/// Find the newest version-suffixed OpenVINO C API library in one configured
/// runtime directory. OpenVINO wheels ship names such as
/// `libopenvino_c.2641.dylib`, while the binding's finder currently searches
/// only for `libopenvino_c.dylib` on macOS.
fn find_versioned_openvino_c_library(directory: &Path) -> Option<PathBuf> {
    let prefix = format!("{}openvino_c", std::env::consts::DLL_PREFIX);
    let suffix = std::env::consts::DLL_SUFFIX;
    let mut candidates = fs::read_dir(directory)
        .ok()?
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let name = path.file_name()?.to_str()?;
            let version = versioned_openvino_library_version(name, &prefix, suffix)?;
            Some((version, path))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    candidates.pop().map(|(_, path)| path)
}

fn versioned_openvino_library_version(
    file_name: &str,
    library_prefix: &str,
    library_suffix: &str,
) -> Option<Vec<u64>> {
    let name_tail = file_name.strip_prefix(library_prefix)?.strip_prefix('.')?;
    let version = if let Some(version) = name_tail.strip_suffix(library_suffix) {
        version
    } else {
        let suffix_without_leading_dot = library_suffix.strip_prefix('.')?;
        name_tail
            .strip_prefix(suffix_without_leading_dot)?
            .strip_prefix('.')?
    };
    if version.is_empty() {
        return None;
    }
    version.split('.').map(|part| part.parse().ok()).collect()
}

impl OpenVinoSession {
    fn new(onnx_path: &Path) -> Result<Self> {
        let onnx_path = onnx_path
            .to_str()
            .context("OpenVINO requires a UTF-8 ONNX model path")?;
        load_openvino_c_library()?;
        let mut core = Core::new().context(
            "failed to load OpenVINO Runtime; install the matching OpenVINO runtime and set its platform library path",
        )?;
        let available_devices = core.available_devices().context(
            "failed to enumerate OpenVINO devices; GPU inference requires the OpenVINO GPU plugin",
        )?;
        let has_gpu = available_devices
            .iter()
            .any(|device| is_openvino_gpu_device(device.as_ref()));
        ensure!(
            has_gpu,
            "OpenVINO GPU inference was selected, but no GPU device is available; available devices: {:?}",
            available_devices
                .iter()
                .map(|device| device.as_ref())
                .collect::<Vec<_>>()
        );
        let model = core
            .read_model_from_file(onnx_path, "")
            .with_context(|| format!("OpenVINO could not read ONNX model {onnx_path}"))?;
        let mut compiled_model = core.compile_model(&model, DeviceType::GPU).context(
            "OpenVINO could not compile the ONNX model for GPU; no CPU fallback is configured",
        )?;
        let execution_devices = compiled_model
            .get_property(&PropertyKey::Other(Cow::Borrowed("EXECUTION_DEVICES")))
            .context("OpenVINO GPU compiled model did not expose EXECUTION_DEVICES")?;
        let execution_devices = parse_openvino_execution_devices(&execution_devices)?;
        ensure!(
            compiled_model.get_input_size()? == 1,
            "OpenVINO model must have one input"
        );
        ensure!(
            compiled_model.get_output_size()? == 1,
            "OpenVINO model must have one output"
        );
        let input = compiled_model.get_input()?;
        let output = compiled_model.get_output()?;
        ensure!(
            input.get_element_type()? == ElementType::F32,
            "OpenVINO ONNX input must be float32"
        );
        ensure!(
            output.get_element_type()? == ElementType::F32,
            "OpenVINO ONNX output must be float32"
        );
        let input_name = input.get_name()?;
        let output_name = output.get_name()?;
        let input_shape = input.get_shape()?.get_dimensions().to_vec();
        let output_shape = output.get_shape()?.get_dimensions().to_vec();
        let request = compiled_model
            .create_infer_request()
            .context("failed to create OpenVINO inference request")?;
        let input_tensor = request
            .get_input_tensor()
            .context("failed to create OpenVINO input tensor")?;
        Ok(Self {
            request,
            input_tensor,
            input_name,
            output_name,
            input_shape,
            output_shape,
            execution_devices,
            _compiled_model: compiled_model,
            _core: core,
        })
    }
}

fn is_openvino_gpu_device(device: &str) -> bool {
    device == "GPU" || device.starts_with("GPU.")
}

fn parse_openvino_execution_devices(value: &str) -> Result<Vec<String>> {
    let devices = value
        .split(',')
        .map(str::trim)
        .filter(|device| !device.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        !devices.is_empty(),
        "OpenVINO GPU compiled model reported an empty EXECUTION_DEVICES value"
    );
    ensure!(
        devices.iter().all(|device| is_openvino_gpu_device(device)),
        "OpenVINO GPU backend reported unexpected execution devices {devices:?}; CPU fallback or a non-GPU device is not permitted"
    );
    Ok(devices)
}

fn validate_session(
    session: &InferenceSession,
    image_size: u32,
    class_names: &[String],
) -> Result<Vec<i64>> {
    let (input_name, input_shape, output_name, output_shape) = match session {
        InferenceSession::Ort(session) => {
            ensure!(
                session.inputs().len() == 1,
                "expected one ONNX input, found {}",
                session.inputs().len()
            );
            ensure!(
                session.outputs().len() == 1,
                "expected one ONNX output, found {}",
                session.outputs().len()
            );
            let input = &session.inputs()[0];
            let output = &session.outputs()[0];
            (
                input.name().to_owned(),
                tensor_shape(input.dtype(), "input")?,
                output.name().to_owned(),
                tensor_shape(output.dtype(), "output")?,
            )
        }
        InferenceSession::Openvino(session) => (
            session.input_name.clone(),
            session.input_shape.clone(),
            session.output_name.clone(),
            session.output_shape.clone(),
        ),
    };
    ensure!(
        input_name == INPUT_NAME,
        "unexpected ONNX input name: {input_name}"
    );
    ensure!(
        output_name == OUTPUT_NAME,
        "unexpected ONNX output name: {output_name}"
    );

    let expected_input = vec![1_i64, 3, image_size as i64, image_size as i64];
    ensure!(
        input_shape == expected_input,
        "unexpected ONNX input shape {input_shape:?}; expected {expected_input:?}"
    );
    let expected_channels = (4 + class_names.len()) as i64;
    let expected_output = vec![
        1_i64,
        expected_channels,
        expected_anchor_count(image_size)? as i64,
    ];
    ensure!(
        output_shape == expected_output,
        "unexpected ONNX output shape {output_shape:?}; expected {expected_output:?}"
    );
    Ok(output_shape)
}

fn tensor_shape(value_type: &ValueType, role: &str) -> Result<Vec<i64>> {
    match value_type {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } => Ok(shape.iter().copied().collect()),
        ValueType::Tensor { ty, .. } => bail!("ONNX {role} tensor must be float32, found {ty:?}"),
        other => bail!("ONNX {role} must be a tensor, found {other:?}"),
    }
}

fn expected_anchor_count(image_size: u32) -> Result<usize> {
    ensure!(
        image_size > 0 && image_size.is_multiple_of(32),
        "image size must be a positive multiple of 32"
    );
    let size = image_size as usize;
    let stride_8 = (size / 8).checked_mul(size / 8);
    let stride_16 = (size / 16).checked_mul(size / 16);
    let stride_32 = (size / 32).checked_mul(size / 32);
    stride_8
        .and_then(|first| stride_16.and_then(|second| first.checked_add(second)))
        .and_then(|sum| stride_32.and_then(|third| sum.checked_add(third)))
        .context("image size overflows YOLO anchor count")
}

fn profile_provider(
    environment: &ort::environment::Environment,
    options: &AuditOptions,
    class_names: &[String],
    expected_output_shape: &[i64],
    sample: PreparedInput,
) -> Result<(ProviderEvidence, f64, f64, PreparedInput)> {
    if options.provider == ProviderSelection::Openvino {
        let init_start = Instant::now();
        let mut session = create_session(environment, options, None)?;
        let output_shape = validate_session(&session, options.image_size, class_names)?;
        ensure!(
            output_shape == expected_output_shape,
            "OpenVINO output signature is unexpected: {output_shape:?}"
        );
        let session_init_ms = elapsed_ms(init_start);
        let execution_devices = match &session {
            InferenceSession::Openvino(session) => session.execution_devices.clone(),
            _ => unreachable!("native OpenVINO session construction returned above"),
        };
        let inference_start = Instant::now();
        let (actual_output_shape, _) =
            run_session(&mut session, &sample.values, options.image_size as usize)?;
        let inference_ms = elapsed_ms(inference_start);
        ensure!(
            actual_output_shape == expected_output_shape,
            "OpenVINO produced unexpected output shape: {actual_output_shape:?}"
        );
        return Ok((
            ProviderEvidence {
                observed_providers: execution_devices,
                event_count: 0,
                node_counts: BTreeMap::new(),
            },
            session_init_ms,
            inference_ms,
            sample,
        ));
    }

    let temporary_directory =
        tempfile::tempdir().context("failed to create ORT profile scratch directory")?;
    let profile_path = temporary_directory.path().join("provider-profile.json");

    let init_start = Instant::now();
    let mut session = create_session(environment, options, Some(&profile_path))?;
    let output_shape = validate_session(&session, options.image_size, class_names)?;
    ensure!(
        output_shape == expected_output_shape,
        "provider probe output signature is unexpected: {output_shape:?}"
    );
    let session_init_ms = elapsed_ms(init_start);

    let inference_start = Instant::now();
    let (output_shape, _) = run_session(&mut session, &sample.values, options.image_size as usize)?;
    let inference_ms = elapsed_ms(inference_start);
    ensure!(
        output_shape == expected_output_shape,
        "provider probe produced unexpected output shape: {output_shape:?}"
    );
    let InferenceSession::Ort(mut session) = session else {
        unreachable!("native OpenVINO profiling returned above")
    };
    let profile_path = session
        .end_profiling()
        .context("failed to finalize ONNX Runtime profile")?;
    let profile_path = PathBuf::from(profile_path);
    let profile = fs::read(&profile_path).with_context(|| {
        format!(
            "failed to read ONNX Runtime profile {}",
            profile_path.display()
        )
    })?;
    let (observed_providers, event_count, node_counts) = parse_provider_profile(&profile)?;

    Ok((
        ProviderEvidence {
            observed_providers,
            event_count,
            node_counts,
        },
        session_init_ms,
        inference_ms,
        sample,
    ))
}

fn parse_provider_profile(profile: &[u8]) -> Result<(Vec<String>, usize, BTreeMap<String, usize>)> {
    let document: Value =
        serde_json::from_slice(profile).context("ONNX Runtime profile is not valid JSON")?;
    let events = document
        .as_array()
        .context("ONNX Runtime profile must be a JSON event array")?;
    let mut providers = BTreeSet::new();
    let mut nodes_by_provider: BTreeMap<String, HashSet<String>> = BTreeMap::new();
    let mut event_count = 0;
    for event in events {
        let Some(args) = event.get("args").and_then(Value::as_object) else {
            continue;
        };
        let Some(provider) = args.get("provider").and_then(Value::as_str) else {
            continue;
        };
        providers.insert(provider.to_owned());
        event_count += 1;
        if let Some(node) = event.get("name").and_then(Value::as_str) {
            nodes_by_provider
                .entry(provider.to_owned())
                .or_default()
                .insert(node.to_owned());
        }
    }
    let node_counts = nodes_by_provider
        .into_iter()
        .map(|(provider, nodes)| (provider, nodes.len()))
        .collect();
    Ok((providers.into_iter().collect(), event_count, node_counts))
}

fn run_session(
    session: &mut InferenceSession,
    values: &[f32],
    image_size: usize,
) -> Result<(Vec<i64>, Vec<f32>)> {
    match session {
        InferenceSession::Ort(session) => {
            let tensor =
                TensorRef::<f32>::from_array_view(([1_usize, 3, image_size, image_size], values))
                    .context("failed to create ONNX input tensor view")?;
            let outputs = session
                .run(ort::inputs! { INPUT_NAME => tensor })
                .context("ONNX Runtime session execution failed")?;
            ensure!(
                outputs.len() == 1,
                "ONNX Runtime returned {} outputs; expected one",
                outputs.len()
            );
            let output = outputs
                .get(OUTPUT_NAME)
                .context("ONNX Runtime did not return the named output")?;
            let (shape, data) = output
                .try_extract_tensor::<f32>()
                .context("ONNX output dtype is not float32")?;
            let shape: Vec<i64> = shape.iter().copied().collect();
            Ok((shape, data.to_vec()))
        }
        InferenceSession::Openvino(session) => {
            let input = session
                .input_tensor
                .get_data_mut::<f32>()
                .context("failed to access OpenVINO input tensor")?;
            ensure!(
                input.len() == values.len(),
                "prepared OpenVINO input has {} values, expected {}",
                values.len(),
                input.len()
            );
            input.copy_from_slice(values);
            session
                .request
                .set_input_tensor(&session.input_tensor)
                .context("failed to bind OpenVINO input tensor")?;
            session
                .request
                .infer()
                .context("OpenVINO GPU inference failed")?;
            let output = session
                .request
                .get_output_tensor()
                .context("failed to retrieve OpenVINO output tensor")?;
            ensure!(
                output.get_element_type()? == ElementType::F32,
                "OpenVINO output tensor dtype changed from float32"
            );
            let shape = output.get_shape()?.get_dimensions().to_vec();
            let values = output
                .get_data::<f32>()
                .context("failed to read OpenVINO output tensor")?
                .to_vec();
            Ok((shape, values))
        }
    }
}

fn prepare_decoded_image(image: RgbImage, decode_ms: f64, image_size: u32) -> Result<TimedInput> {
    let preprocess_start = Instant::now();
    let prepared = prepare_rgb_image(&image, image_size)?;
    let preprocess_ms = elapsed_ms(preprocess_start);
    Ok(TimedInput {
        image,
        prepared,
        decode_ms,
        preprocess_ms,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .with_context(|| format!("failed to hash {}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn summarize_calibration_counts(
    options: &CalibrationAuditOptions,
    applied_image_space: ImageSpace,
    frame_count: usize,
    calibrated_frame_count: usize,
    calibrated_detection_count: usize,
) -> CalibrationAuditReport {
    let (bundle_source_width_px, bundle_source_height_px) =
        options.calibration.source_dimensions_px();
    CalibrationAuditReport {
        bundle_path: options.bundle_path.display().to_string(),
        bundle_sha256: options.bundle_sha256.clone(),
        bundle_calibration_revision: options.calibration.calibration_revision(),
        bundle_camera_profile_id: options.calibration.camera_profile_id().to_owned(),
        bundle_rectification_revision: options.calibration.rectification_revision(),
        bundle_image_space: options.calibration.image_space(),
        bundle_source_width_px,
        bundle_source_height_px,
        applied_calibration_revision: options.applied_calibration_revision,
        applied_camera_profile_id: options.applied_camera_profile_id.clone(),
        applied_rectification_revision: options.applied_rectification_revision,
        applied_image_space,
        frame_count,
        calibrated_frame_count,
        unavailable_frame_count: frame_count - calibrated_frame_count,
        calibrated_detection_count,
    }
}

fn calibration_latency_summary(
    mode: VideoReplayMode,
    calibration_enabled: bool,
    samples_ms: Vec<f64>,
) -> Option<LatencyPercentiles> {
    (mode == VideoReplayMode::RealtimeLatestFrame && calibration_enabled)
        .then(|| percentile_summary(samples_ms))
}

fn paced_source_to_calibration_latency_ms(
    scheduled_at: Option<Instant>,
    source_arrived_at: Option<Instant>,
    calibration_completed_at: Instant,
) -> Option<f64> {
    scheduled_at
        .zip(source_arrived_at)
        .map(|(_, source_arrived_at)| {
            calibration_completed_at
                .saturating_duration_since(source_arrived_at)
                .as_secs_f64()
                * 1000.0
        })
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::{
        build_detection_batch, calibration_latency_summary, calibration_report_for_batch,
        checked_duration_from_seconds, completed_after_source_deadline, expected_anchor_count,
        find_versioned_openvino_c_library, is_openvino_gpu_device, latest_due_source_index,
        load_and_verify_manifest, open_detection_batch_sink, open_scene_snapshot_output,
        paced_source_to_calibration_latency_ms, parse_openvino_execution_devices,
        parse_provider_profile, percentile_summary, run_bounded_command, sha256_file,
        timeline_media_duration, validate_batch_identifier, validate_frame_inference_options,
        validate_inference_parameters, validate_scene_calibration_context,
        versioned_openvino_library_version, write_detection_batch_line, write_scene_snapshot_line,
        AuditOptions, BatchOutputOptions, CalibrationAuditOptions, CalibrationStatus,
        FrameInferenceEngine, FrameInferenceOptions, PacedReplayMeasurementWindow,
        ProviderSelection, SceneSnapshotOutputDestination, SceneSnapshotOutputOptions,
        VideoReplayMode, VideoTimeline,
    };
    use crate::ball_localization_profile::BallLocalizationSettings;
    use crate::detection_batch::{decode_detection_batch_v1, DetectionBatchV1, ImageSpace};
    use crate::inference::Detection;
    use crate::live_timing_spans::FrameTimingContext;
    use crate::tracking::TrackingConfigV1;
    use image::{Rgb, RgbImage};
    use serde_json::{json, Value};
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    #[cfg(unix)]
    use std::process::Command;
    use std::time::{Duration, Instant};

    const SUCCESS_BATCH_FIXTURE: &str =
        include_str!("../../contracts/detection-batch/v1/detection-batch-success.json");
    const EMPTY_BATCH_FIXTURE: &str =
        include_str!("../../contracts/detection-batch/v1/detection-batch-empty-success.json");
    const TRACKED_SCENE_FIXTURE: &str =
        include_str!("../../contracts/v1/scene-rust-tracked-success.json");

    fn batch_options() -> (AuditOptions, BatchOutputOptions) {
        (
            AuditOptions {
                onnx_path: PathBuf::from("model.onnx"),
                source_checkpoint_path: PathBuf::from("source.pt"),
                input_path: PathBuf::from("frames"),
                provider: ProviderSelection::Cpu,
                image_size: 1280,
                confidence: 0.25,
                iou: 0.45,
                max_detections: 300,
                intra_op_threads: None,
                warmup_runs: 1,
                source_image_space: ImageSpace::Raw,
                detection_batch_output: None,
                calibration: None,
                ball_localization_settings: BallLocalizationSettings::default(),
                scene_snapshot_output: None,
            },
            BatchOutputOptions {
                jsonl_path: PathBuf::from("batches.jsonl"),
                engine_instance_id: "engine-test-01".to_owned(),
                source_session_id: "recording-test-01".to_owned(),
            },
        )
    }

    fn calibration_options(calibration_revision: u64) -> CalibrationAuditOptions {
        let bundle_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("contracts/calibration/v1/table-calibration-center-origin.json");
        CalibrationAuditOptions::load(bundle_path, "test-camera-profile", calibration_revision, 2)
            .unwrap()
    }

    fn calibrated_batch(source: &str) -> DetectionBatchV1 {
        let mut value: Value = serde_json::from_str(source).unwrap();
        value["source"]["width_px"] = json!(1000);
        value["source"]["height_px"] = json!(500);
        value["image_space"] = json!("rectified");
        if let Some(detection) = value["detections"]
            .as_array_mut()
            .and_then(|items| items.first_mut())
        {
            detection["bbox_image_px"] = json!({
                "left": 400.0,
                "top": 200.0,
                "right": 600.0,
                "bottom": 300.0
            });
            detection["center_image_px"] = json!({"x": 500.0, "y": 250.0});
        }
        decode_detection_batch_v1(&serde_json::to_vec(&value).unwrap()).unwrap()
    }

    #[test]
    fn cloth_geometry_report_never_emits_ball_center_table_coordinates() {
        let populated_batch = calibrated_batch(SUCCESS_BATCH_FIXTURE);
        let valid =
            calibration_report_for_batch(&populated_batch, &calibration_options(3)).unwrap();
        assert!(matches!(valid.status, CalibrationStatus::Valid));
        assert_eq!(
            valid.projected_box_midpoint_count,
            Some(populated_batch.detections.len())
        );

        let unavailable =
            calibration_report_for_batch(&populated_batch, &calibration_options(4)).unwrap();
        assert!(matches!(unavailable.status, CalibrationStatus::Unavailable));
        assert!(unavailable
            .reason
            .as_deref()
            .unwrap()
            .contains("CalibrationRevisionMismatch"));
        assert!(unavailable.projected_box_midpoint_count.is_none());

        let empty_batch = calibrated_batch(EMPTY_BATCH_FIXTURE);
        let empty = calibration_report_for_batch(&empty_batch, &calibration_options(3)).unwrap();
        assert!(matches!(empty.status, CalibrationStatus::Valid));
        assert_eq!(empty.projected_box_midpoint_count, Some(0));
    }

    #[test]
    fn calibration_latency_is_reported_only_for_paced_source_timing() {
        let source_arrived_at = Instant::now();
        let completed_at = source_arrived_at + Duration::from_millis(12);

        assert_eq!(
            paced_source_to_calibration_latency_ms(
                Some(source_arrived_at),
                Some(source_arrived_at),
                completed_at,
            ),
            Some(12.0)
        );
        assert!(paced_source_to_calibration_latency_ms(
            None,
            Some(source_arrived_at),
            completed_at,
        )
        .is_none());
        assert!(paced_source_to_calibration_latency_ms(
            Some(source_arrived_at),
            None,
            completed_at,
        )
        .is_none());

        let realtime =
            calibration_latency_summary(VideoReplayMode::RealtimeLatestFrame, true, vec![12.0])
                .unwrap();
        assert_eq!(realtime.sample_count, 1);
        assert!(calibration_latency_summary(VideoReplayMode::Capacity, true, vec![12.0]).is_none());
        assert!(calibration_latency_summary(
            VideoReplayMode::RealtimeLatestFrame,
            false,
            vec![12.0],
        )
        .is_none());
    }

    #[test]
    fn calibration_audit_options_reject_unsafe_applied_context() {
        let bundle_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("contracts/calibration/v1/table-calibration-center-origin.json");
        assert!(CalibrationAuditOptions::load(bundle_path.clone(), "", 3, 2).is_err());
        assert!(CalibrationAuditOptions::load(
            bundle_path,
            "test-camera-profile",
            super::MAX_SAFE_JSON_INTEGER + 1,
            2
        )
        .is_err());
    }

    #[test]
    fn scene_tracking_and_applied_calibration_must_share_camera_geometry_identity() {
        let calibration = calibration_options(3);
        let matching = TrackingConfigV1::new(
            1,
            Some(0),
            "test-camera-profile",
            12.0,
            Duration::from_millis(50),
            Duration::from_millis(200),
            Duration::from_millis(1000),
            24,
            32,
            1,
            2,
        )
        .unwrap();
        assert!(validate_scene_calibration_context(&matching, &calibration).is_ok());

        let wrong_profile = TrackingConfigV1::new(
            1,
            Some(0),
            "other-camera-profile",
            12.0,
            Duration::from_millis(50),
            Duration::from_millis(200),
            Duration::from_millis(1000),
            24,
            32,
            1,
            2,
        )
        .unwrap();
        assert!(validate_scene_calibration_context(&wrong_profile, &calibration).is_err());

        let wrong_rectification = TrackingConfigV1::new(
            1,
            Some(0),
            "test-camera-profile",
            12.0,
            Duration::from_millis(50),
            Duration::from_millis(200),
            Duration::from_millis(1000),
            24,
            32,
            1,
            3,
        )
        .unwrap();
        assert!(validate_scene_calibration_context(&wrong_rectification, &calibration).is_err());
    }

    #[test]
    fn output_anchor_count_matches_yolov8_stride_heads() {
        assert_eq!(expected_anchor_count(1280).unwrap(), 33_600);
        assert_eq!(expected_anchor_count(64).unwrap(), 84);
        assert!(expected_anchor_count(65).is_err());
    }

    #[test]
    fn video_latest_frame_schedule_keeps_the_newest_due_source_index() {
        let pts = [0.0, 0.04, 0.08, 0.12, 0.20];
        assert_eq!(latest_due_source_index(&pts, 0, 0.0), 0);
        assert_eq!(latest_due_source_index(&pts, 1, 0.079), 1);
        assert_eq!(latest_due_source_index(&pts, 1, 0.08), 2);
        assert_eq!(latest_due_source_index(&pts, 1, 0.19), 3);
        assert_eq!(latest_due_source_index(&pts, 4, 0.19), 4);
        assert_eq!(latest_due_source_index(&pts, 4, 0.20), 4);
    }

    #[test]
    fn video_media_duration_uses_median_frame_interval_for_the_final_frame() {
        let timeline = VideoTimeline {
            stream_index: 0,
            width: 640,
            height: 480,
            origin_seconds: 10.0,
            presentation_times_seconds: vec![0.0, 0.04, 0.08, 0.20],
        };
        assert!((timeline_media_duration(&timeline).unwrap() - 0.24).abs() < 1e-12);
        assert_eq!(
            timeline_media_duration(&VideoTimeline {
                presentation_times_seconds: vec![0.0],
                ..timeline
            }),
            None
        );
    }

    #[test]
    fn video_duration_conversion_rejects_non_finite_and_preserves_subseconds() {
        assert_eq!(
            checked_duration_from_seconds(1.25).unwrap(),
            Duration::from_millis(1250)
        );
        assert!(checked_duration_from_seconds(f64::NAN).is_err());
        assert!(checked_duration_from_seconds(f64::INFINITY).is_err());
    }

    #[test]
    fn processed_source_deadline_is_a_half_open_completion_boundary() {
        let source_start = Instant::now();
        let completed_on_time = source_start + Duration::from_millis(40);
        let completed_late = completed_on_time + Duration::from_nanos(1);
        assert!(!completed_after_source_deadline(source_start, 0.04, completed_on_time).unwrap());
        assert!(completed_after_source_deadline(source_start, 0.04, completed_late).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_child_process_enforces_output_limit_timeout_and_exit_status() {
        let overflow = Command::new("sh");
        let mut overflow = overflow;
        overflow.arg("-c").arg("printf 123456");
        assert!(
            run_bounded_command(overflow, Duration::from_secs(1), 5, "test output")
                .unwrap_err()
                .to_string()
                .contains("size bound")
        );

        let mut timeout = Command::new("sh");
        timeout.arg("-c").arg("exec /bin/sleep 1");
        assert!(
            run_bounded_command(timeout, Duration::from_millis(30), 16, "test timeout")
                .unwrap_err()
                .to_string()
                .contains("timeout")
        );

        let mut failed = Command::new("sh");
        failed.arg("-c").arg("printf failure >&2; exit 7");
        assert!(
            run_bounded_command(failed, Duration::from_secs(1), 16, "test exit")
                .unwrap_err()
                .to_string()
                .contains("failure")
        );
    }

    #[cfg(unix)]
    #[test]
    fn bounded_video_decoder_rejects_short_and_extra_output() {
        let directory = tempfile::tempdir().unwrap();
        let fake_ffmpeg = directory.path().join("fake-ffmpeg.sh");
        fs::write(&fake_ffmpeg, "#!/bin/sh\nprintf 123456\n").unwrap();
        let mut permissions = fs::metadata(&fake_ffmpeg).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_ffmpeg, permissions).unwrap();

        let short = super::decode_video_to_spool(
            &fake_ffmpeg,
            PathBuf::from("ignored-input.mp4").as_path(),
            0,
            9,
            32,
            Duration::from_secs(1),
        );
        let short_error = short.unwrap_err().to_string();
        assert!(
            short_error.contains("expected 9 bytes"),
            "unexpected short decode error: {short_error}"
        );

        let extra = super::decode_video_to_spool(
            &fake_ffmpeg,
            PathBuf::from("ignored-input.mp4").as_path(),
            0,
            5,
            32,
            Duration::from_secs(1),
        );
        assert!(extra.unwrap_err().to_string().contains("byte limit"));
    }

    #[test]
    fn paced_measurement_window_has_fixed_half_open_boundaries() {
        let start = Instant::now();
        let window = PacedReplayMeasurementWindow::new(start, Duration::from_millis(100)).unwrap();

        assert!(window.contains(start));
        assert!(window.contains(start + Duration::from_millis(99)));
        assert!(!window.contains(window.end));
        assert!(!window.contains(window.end + Duration::from_nanos(1)));
        assert_eq!(window.duration(), Duration::from_millis(100));
    }

    #[test]
    fn percentile_summary_uses_nearest_rank_and_reports_empty_samples() {
        let result = percentile_summary(vec![4.0, 1.0, 3.0, 2.0]);
        assert_eq!(result.sample_count, 4);
        assert_eq!(result.p50_ms, Some(2.0));
        assert_eq!(result.p95_ms, Some(4.0));
        assert_eq!(result.p99_ms, Some(4.0));

        let empty = percentile_summary(Vec::new());
        assert_eq!(empty.sample_count, 0);
        assert_eq!(empty.p50_ms, None);
    }

    #[test]
    fn provider_name_is_observed_only_from_ort_node_profile_events() {
        let trace = br#"[
          {"name":"Conv_kernel_time","cat":"Node","args":{"provider":"CPUExecutionProvider"}},
          {"name":"Conv_kernel_time","cat":"Node","args":{"provider":"CPUExecutionProvider"}},
          {"name":"Graph","cat":"Session","args":{}}
        ]"#;
        let (providers, events, counts) = parse_provider_profile(trace).unwrap();
        assert_eq!(providers, ["CPUExecutionProvider"]);
        assert_eq!(events, 2);
        assert_eq!(
            counts,
            BTreeMap::from([("CPUExecutionProvider".to_owned(), 1)])
        );
    }

    #[test]
    fn provider_is_unavailable_when_trace_has_no_execution_nodes() {
        let (providers, events, counts) = parse_provider_profile(br#"[{"args":{}}]"#).unwrap();
        assert!(providers.is_empty());
        assert_eq!(events, 0);
        assert!(counts.is_empty());
    }

    #[test]
    fn manifest_requires_matching_source_and_export_hashes() {
        let directory = tempfile::tempdir().unwrap();
        let onnx_path = directory.path().join("model.onnx");
        let source_path = directory.path().join("source.pt");
        fs::write(&onnx_path, b"onnx test bytes").unwrap();
        fs::write(&source_path, b"source checkpoint test bytes").unwrap();
        let manifest = serde_json::json!({
            "manifest_schema_version": 1,
            "source_model": { "sha256": super::sha256_file(&source_path).unwrap() },
            "exported_model": {
                "path": "model.onnx",
                "sha256": super::sha256_file(&onnx_path).unwrap()
            },
            "model": {
                "class_count": 2,
                "class_names": [
                    { "id": 0, "name": "cue" },
                    { "id": 1, "name": "ball" }
                ]
            }
        });
        let manifest_path = PathBuf::from(format!("{}.manifest.json", onnx_path.display()));
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

        let loaded = load_and_verify_manifest(&onnx_path, &source_path).unwrap();
        assert_eq!(loaded.class_names, ["cue", "ball"]);

        fs::write(&source_path, b"different source checkpoint").unwrap();
        let error = load_and_verify_manifest(&onnx_path, &source_path).unwrap_err();
        assert!(error.to_string().contains("source checkpoint hash"));

        fs::write(&source_path, b"source checkpoint test bytes").unwrap();
        fs::write(&onnx_path, b"different ONNX artifact").unwrap();
        let error = load_and_verify_manifest(&onnx_path, &source_path).unwrap_err();
        assert!(error.to_string().contains("exported ONNX hash"));
    }

    #[test]
    fn manifest_v1_contract_fixture_is_accepted_and_unknown_versions_are_rejected() {
        let fixture_directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts/model-export/v1");
        let temporary = tempfile::tempdir().unwrap();
        let model_path = temporary.path().join("billiards-yolov8n.onnx");
        let source_path = temporary.path().join("billiards-yolov8n.pt");
        fs::copy(
            fixture_directory.join("exported-model.fixture"),
            &model_path,
        )
        .unwrap();
        fs::copy(
            fixture_directory.join("source-checkpoint.fixture"),
            &source_path,
        )
        .unwrap();
        let manifest_path = PathBuf::from(format!("{}.manifest.json", model_path.display()));
        fs::copy(
            fixture_directory.join("manifest-contract.json"),
            &manifest_path,
        )
        .unwrap();

        let loaded = load_and_verify_manifest(&model_path, &source_path).unwrap();
        assert_eq!(loaded.class_names, ["cue", "ball"]);
        assert_eq!(loaded.artifact_sha256, sha256_file(&model_path).unwrap());
        assert_eq!(
            loaded.source_checkpoint_sha256,
            sha256_file(&source_path).unwrap()
        );

        let mut unsupported: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap())
            .expect("shared manifest fixture is valid JSON");
        unsupported["manifest_schema_version"] = serde_json::json!(2);
        fs::write(&manifest_path, serde_json::to_vec(&unsupported).unwrap()).unwrap();
        let error = load_and_verify_manifest(&model_path, &source_path).unwrap_err();
        assert!(error
            .to_string()
            .contains("unsupported ONNX export manifest schema version: 2"));
    }

    #[test]
    fn provider_policy_report_names_the_selected_policy() {
        assert_eq!(
            ProviderSelection::Cpu.requested_provider_name(),
            "CPUExecutionProvider"
        );
        assert_eq!(
            ProviderSelection::Auto.requested_provider_name(),
            "AutoDevicePolicy::MaxEfficiency"
        );
        assert_eq!(
            ProviderSelection::Openvino.requested_provider_name(),
            "OpenVINO GPU (native)"
        );
        assert_eq!(
            ProviderSelection::Openvino.binding(),
            (
                "openvino-rs",
                "0.11.1",
                "Native OpenVINO GPU backend explicitly selected; runtime EXECUTION_DEVICES is checked to confirm GPU execution, and per-node assignment is unavailable"
            )
        );
        assert_eq!(
            ProviderSelection::Cpu.binding().0,
            "ort",
            "portable CPU inference remains on ONNX Runtime"
        );
    }

    #[test]
    fn openvino_execution_device_checks_require_gpu_only() {
        assert!(is_openvino_gpu_device("GPU"));
        assert!(is_openvino_gpu_device("GPU.0"));
        assert!(!is_openvino_gpu_device("CPU"));
        assert!(!is_openvino_gpu_device("GPU_FALLBACK"));

        assert_eq!(
            parse_openvino_execution_devices("GPU.0, GPU.1").unwrap(),
            ["GPU.0", "GPU.1"]
        );
        assert_eq!(parse_openvino_execution_devices("GPU").unwrap(), ["GPU"]);
        assert!(parse_openvino_execution_devices("").is_err());
        assert!(parse_openvino_execution_devices("CPU").is_err());
        assert!(parse_openvino_execution_devices("GPU.0, CPU").is_err());
    }

    #[test]
    fn versioned_openvino_library_discovery_handles_platform_naming_and_orders_versions() {
        let scratch = tempfile::tempdir().unwrap();
        let prefix = format!("{}openvino_c", std::env::consts::DLL_PREFIX);
        let suffix = std::env::consts::DLL_SUFFIX;
        let library_name = |version: &str| {
            if suffix == ".so" {
                format!("{prefix}{suffix}.{version}")
            } else {
                format!("{prefix}.{version}{suffix}")
            }
        };
        let older = scratch.path().join(library_name("2025.4.0"));
        let newer = scratch.path().join(library_name("2026.4.1"));
        fs::write(&older, b"older runtime").unwrap();
        fs::write(&newer, b"newer runtime").unwrap();
        fs::write(
            scratch.path().join(library_name("not-a-version")),
            b"invalid runtime name",
        )
        .unwrap();

        assert_eq!(
            find_versioned_openvino_c_library(scratch.path()),
            Some(newer)
        );
        assert_eq!(
            versioned_openvino_library_version(&library_name("2026.4.1"), &prefix, suffix),
            Some(vec![2026, 4, 1])
        );
        assert!(
            versioned_openvino_library_version(&format!("{prefix}{suffix}"), &prefix, suffix)
                .is_none()
        );
    }

    #[test]
    fn batch_builder_preserves_frame_model_settings_and_detection_geometry() {
        let (mut options, output) = batch_options();
        options.source_image_space = ImageSpace::Rectified;
        let origin = Instant::now();
        let detection = Detection {
            class_id: 1,
            class_name: "ball".to_owned(),
            confidence: 0.875,
            bbox_xyxy_px: [10.0, 20.0, 30.0, 40.0],
            center_xy_px: [20.0, 30.0],
        };

        let batch = build_detection_batch(
            2,
            3,
            1920,
            1080,
            &[detection],
            &["cue".to_owned(), "ball".to_owned()],
            &"a".repeat(64),
            &"b".repeat(64),
            &options,
            &output,
            origin,
            origin + Duration::from_nanos(10),
            origin + Duration::from_nanos(20),
            None,
        )
        .unwrap();

        assert_eq!(batch.batch_sequence, 3);
        assert_eq!(batch.source.frame_id, "frame-000003");
        assert_eq!(
            (batch.source.width_px, batch.source.height_px),
            (1920, 1080)
        );
        assert!(batch.source.timestamp.is_none());
        assert_eq!(batch.model.classes[1].class_name, "ball");
        assert_eq!(batch.settings.input_width_px, 1280);
        assert_eq!(batch.image_space, ImageSpace::Rectified);
        assert_eq!(batch.detections.len(), 1);
        assert_eq!(batch.detections[0].detection_index, 0);
        assert_eq!(batch.detections[0].bbox_image_px.left, 10.0);
        assert_eq!(batch.detections[0].center_image_px.y, 30.0);
        assert!(batch.published_at.ticks.parse::<u64>().unwrap() >= 20);
        batch.validate().unwrap();
    }

    #[test]
    fn paced_batch_keeps_source_frame_identity_separate_from_publication_sequence() {
        let (options, output) = batch_options();
        let origin = Instant::now();
        let batch = build_detection_batch(
            7,
            2,
            640,
            480,
            &[],
            &["cue".to_owned(), "ball".to_owned()],
            &"a".repeat(64),
            &"b".repeat(64),
            &options,
            &output,
            origin,
            origin + Duration::from_nanos(10),
            origin + Duration::from_nanos(20),
            Some(origin + Duration::from_nanos(5)),
        )
        .unwrap();

        assert_eq!(batch.batch_sequence, 2);
        assert_eq!(batch.source.frame_id, "frame-000008");
        let source_timestamp = batch.source.timestamp.as_ref().unwrap();
        assert_eq!(
            source_timestamp.stage,
            crate::detection_batch::TimestampStage::FrameArrival
        );
        assert_eq!(source_timestamp.ticks, "5");
        batch.validate().unwrap();
    }

    #[test]
    fn batch_builder_represents_no_detections_as_successful_complete_observation() {
        let (options, output) = batch_options();
        let origin = Instant::now();
        let batch = build_detection_batch(
            0,
            1,
            640,
            480,
            &[],
            &["cue".to_owned(), "ball".to_owned()],
            &"a".repeat(64),
            &"b".repeat(64),
            &options,
            &output,
            origin,
            origin,
            origin,
            None,
        )
        .unwrap();

        assert_eq!(batch.outcome, crate::detection_batch::BatchOutcome::Success);
        assert!(batch.reason.is_none());
        assert!(batch.detections.is_empty());
        batch.validate().unwrap();
    }

    #[test]
    fn jsonl_writer_emits_complete_ordered_contract_records() {
        let (options, output) = batch_options();
        let origin = Instant::now();
        let make_batch = |index| {
            build_detection_batch(
                index,
                index + 1,
                640,
                480,
                &[],
                &["cue".to_owned(), "ball".to_owned()],
                &"a".repeat(64),
                &"b".repeat(64),
                &options,
                &output,
                origin,
                origin + Duration::from_nanos(index as u64 + 1),
                origin + Duration::from_nanos(index as u64 + 2),
                None,
            )
            .unwrap()
        };
        let mut writer = std::io::BufWriter::new(Vec::new());
        write_detection_batch_line(&mut writer, &make_batch(0)).unwrap();
        write_detection_batch_line(&mut writer, &make_batch(1)).unwrap();
        let output = writer.into_inner().unwrap();

        let lines: Vec<_> = output.split(|byte| *byte == b'\n').collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[2].is_empty());
        let first = decode_detection_batch_v1(lines[0]).unwrap();
        let second = decode_detection_batch_v1(lines[1]).unwrap();
        assert_eq!(first.batch_sequence, 1);
        assert_eq!(second.batch_sequence, 2);
        assert_eq!(first.engine_instance_id, second.engine_instance_id);
        assert_eq!(first.source.session_id, second.source.session_id);
    }

    #[test]
    fn scene_jsonl_writer_emits_a_complete_validated_snapshot_line() {
        let snapshot = crate::decode_scene_snapshot_v1(TRACKED_SCENE_FIXTURE.as_bytes()).unwrap();
        let mut expected = crate::encode_scene_snapshot_v1(&snapshot).unwrap();
        expected.push(b'\n');
        let mut bytes = Vec::new();
        write_scene_snapshot_line(&mut bytes, &snapshot).unwrap();
        assert_eq!(
            bytes, expected,
            "raw JSONL bytes remain the v1 snapshot format"
        );
        assert_eq!(bytes.last(), Some(&b'\n'));
        let decoded = crate::decode_scene_snapshot_v1(&bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(decoded, snapshot);
    }

    #[test]
    fn scene_output_creation_never_overwrites_an_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let options = SceneSnapshotOutputOptions {
            destination: SceneSnapshotOutputDestination::JsonlFile {
                path: directory.path().join("scene.jsonl"),
            },
            tracking: TrackingConfigV1::new(
                1,
                None,
                "camera-test",
                12.0,
                Duration::from_millis(50),
                Duration::from_millis(200),
                Duration::from_millis(1000),
                24,
                32,
                1,
                1,
            )
            .unwrap(),
        };
        let output = open_scene_snapshot_output(&options).unwrap();
        drop(output);
        assert!(open_scene_snapshot_output(&options).is_err());
    }

    #[test]
    fn batch_identity_validation_rejects_empty_long_and_control_identifiers() {
        assert!(validate_batch_identifier("", "engine instance ID").is_err());
        assert!(validate_batch_identifier(&"a".repeat(129), "source session ID").is_err());
        assert!(validate_batch_identifier("recording\n1", "source session ID").is_err());
        assert!(validate_batch_identifier("recording-1", "source session ID").is_ok());
    }

    #[test]
    fn batch_output_creation_never_overwrites_an_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("existing.jsonl");
        fs::write(&path, b"preserve this data").unwrap();
        let options = BatchOutputOptions {
            jsonl_path: path.clone(),
            engine_instance_id: "engine-test".to_owned(),
            source_session_id: "session-test".to_owned(),
        };

        assert!(open_detection_batch_sink(&options).is_err());
        assert_eq!(fs::read(path).unwrap(), b"preserve this data");
    }

    #[test]
    fn batch_writer_propagates_sink_failures() {
        struct FailingWriter;

        impl std::io::Write for FailingWriter {
            fn write(&mut self, _buffer: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("injected sink failure"))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let (options, output) = batch_options();
        let origin = Instant::now();
        let batch = build_detection_batch(
            0,
            1,
            640,
            480,
            &[],
            &["cue".to_owned(), "ball".to_owned()],
            &"a".repeat(64),
            &"b".repeat(64),
            &options,
            &output,
            origin,
            origin,
            origin,
            None,
        )
        .unwrap();

        assert!(write_detection_batch_line(&mut FailingWriter, &batch).is_err());
    }

    #[test]
    fn frame_engine_runs_one_rgb_observation_through_the_verified_model_session() {
        let scratch = tempfile::tempdir().unwrap();
        let model_path = scratch.path().join("tiny-two-class.onnx");
        fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/tiny_two_class_detector.onnx"
            ),
            &model_path,
        )
        .unwrap();
        let checkpoint_path = scratch.path().join("source.pt");
        fs::write(&checkpoint_path, b"test checkpoint lineage").unwrap();
        let manifest_path = PathBuf::from(format!("{}.manifest.json", model_path.display()));
        fs::write(
            manifest_path,
            serde_json::to_vec(&json!({
                "manifest_schema_version": 1,
                "source_model": { "sha256": sha256_file(&checkpoint_path).unwrap() },
                "exported_model": {
                    "path": "tiny-two-class.onnx",
                    "sha256": sha256_file(&model_path).unwrap()
                },
                "model": {
                    "class_count": 2,
                    "class_names": [
                        { "id": 0, "name": "cue" },
                        { "id": 1, "name": "ball" }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let environment = ort::init()
            .with_name("billiards-frame-engine-test")
            .with_telemetry(false)
            .build()
            .unwrap();
        let mut engine = FrameInferenceEngine::new(
            environment,
            FrameInferenceOptions {
                onnx_path: Some(model_path.clone()),
                source_checkpoint_path: checkpoint_path,
                provider: ProviderSelection::Cpu,
                image_size: 32,
                confidence: 0.4,
                iou: 0.45,
                max_detections: 300,
                intra_op_threads: Some(1),
            },
        )
        .unwrap();

        let image = RgbImage::from_pixel(32, 32, Rgb([12, 34, 56]));
        let output = engine.infer_rgb(&image).unwrap();
        let borrowed_output = engine
            .infer_rgb8(image.width(), image.height(), image.as_raw())
            .unwrap();
        let timing = FrameTimingContext::new(Instant::now());
        let detector_span = timing.start_span("detector").unwrap();
        let timed_output = engine
            .infer_rgb8_with_spans(
                image.width(),
                image.height(),
                image.as_raw(),
                &timing,
                detector_span.id(),
            )
            .unwrap();
        drop(detector_span);
        let timing_snapshot = timing.snapshot().unwrap();
        let timing_names = timing_snapshot
            .spans
            .iter()
            .map(|span| span.name.as_str())
            .collect::<BTreeSet<_>>();
        assert!(timing_names.contains("detector"));
        assert!(timing_names.contains("preprocess"));
        assert!(timing_names.contains("model_execute"));
        assert!(timing_names.contains("decode_nms"));
        assert_eq!(timed_output.detections, borrowed_output.detections);
        assert_eq!(borrowed_output.source_width_px, output.source_width_px);
        assert_eq!(borrowed_output.source_height_px, output.source_height_px);
        assert_eq!(borrowed_output.detections, output.detections);
        assert!(engine.infer_rgb8(32, 32, &[0; 11]).is_err());
        assert_eq!((output.source_width_px, output.source_height_px), (32, 32));
        assert_eq!(output.detections.len(), 1);
        assert_eq!(output.detections[0].class_name, "ball");
        assert_eq!(output.detections[0].center_xy_px, [16.0, 16.0]);
        assert_eq!(
            engine.model_identity().artifact_sha256,
            sha256_file(&model_path).unwrap()
        );
        assert_eq!(engine.model_identity().classes[1].class_name, "ball");
        assert_eq!(engine.inference_settings().input_width_px, 32);
        assert_eq!(engine.requested_provider(), ProviderSelection::Cpu);
        assert!(output.timings.preprocess_ms.is_finite());
        assert!(output.timings.inference_ms.is_finite());
        assert!(output.timings.decode_and_nms_ms.is_finite());
        assert!(output.timings.engine_total_ms >= output.timings.preprocess_ms);
    }

    #[test]
    fn openvino_live_engine_bypasses_ort_environment_and_keeps_cpu_requirement() {
        let scratch = tempfile::tempdir().unwrap();
        let model_path = scratch.path().join("tiny-two-class.onnx");
        fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/tiny_two_class_detector.onnx"
            ),
            &model_path,
        )
        .unwrap();
        let checkpoint_path = scratch.path().join("source.pt");
        fs::write(&checkpoint_path, b"test checkpoint lineage").unwrap();
        fs::write(
            format!("{}.manifest.json", model_path.display()),
            serde_json::to_vec(&json!({
                "manifest_schema_version": 1,
                "source_model": { "sha256": sha256_file(&checkpoint_path).unwrap() },
                "exported_model": {
                    "path": "tiny-two-class.onnx",
                    "sha256": sha256_file(&model_path).unwrap()
                },
                "model": {
                    "class_count": 2,
                    "class_names": [
                        { "id": 0, "name": "cue" },
                        { "id": 1, "name": "ball" }
                    ]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let make_options = |provider| FrameInferenceOptions {
            onnx_path: Some(model_path.clone()),
            source_checkpoint_path: checkpoint_path.clone(),
            provider,
            image_size: 32,
            confidence: 0.4,
            iou: 0.45,
            max_detections: 300,
            intra_op_threads: Some(1),
        };

        let cpu_error = match FrameInferenceEngine::new_without_onnx_runtime(make_options(
            ProviderSelection::Cpu,
        )) {
            Ok(_) => {
                panic!("CPU provider unexpectedly constructed without an ONNX Runtime environment")
            }
            Err(error) => format!("{error:#}"),
        };
        assert!(cpu_error.contains("ONNX Runtime environment is required"));

        match FrameInferenceEngine::new_without_onnx_runtime(make_options(
            ProviderSelection::Openvino,
        )) {
            Ok(engine) => assert_eq!(engine.requested_provider(), ProviderSelection::Openvino),
            Err(error) => {
                let message = format!("{error:#}");
                assert!(
                    !message.contains("ONNX Runtime environment is required"),
                    "native OpenVINO failure was incorrectly reported as an ONNX Runtime requirement: {message}"
                );
                assert!(
                    message.contains("OpenVINO") || message.contains("GPU"),
                    "native OpenVINO construction returned an unrelated error: {message}"
                );
            }
        }
    }

    #[test]
    fn live_frame_options_share_the_audit_inference_parameter_validation() {
        assert!(validate_inference_parameters(32, 0.4, 0.45, 1, Some(1)).is_ok());
        assert!(validate_inference_parameters(31, 0.4, 0.45, 1, Some(1)).is_err());
        assert!(validate_inference_parameters(32, f32::NAN, 0.45, 1, Some(1)).is_err());
        assert!(validate_inference_parameters(32, 0.4, 0.45, 0, Some(1)).is_err());
        assert!(validate_inference_parameters(32, 0.4, 0.45, 1, Some(0)).is_err());

        let scratch = tempfile::tempdir().unwrap();
        let options = FrameInferenceOptions {
            onnx_path: Some(scratch.path().join("missing.onnx")),
            source_checkpoint_path: scratch.path().join("missing.pt"),
            provider: ProviderSelection::Cpu,
            image_size: 32,
            confidence: 0.4,
            iou: 0.45,
            max_detections: 1,
            intra_op_threads: None,
        };
        assert!(validate_frame_inference_options(&options).is_err());
    }
}
