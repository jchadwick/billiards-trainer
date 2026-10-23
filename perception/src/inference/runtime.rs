use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use image::{ImageReader, RgbImage};
use openvino::{Core, DeviceType, ElementType, InferRequest, RwPropertyKey, Tensor};
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

use crate::detection_batch::{
    encode_detection_batch_v1, BatchOutcome, BatchTimestamp, BoundingBox, ClockKind,
    Detection as BatchDetection, DetectionBatchV1, ImageSpace, InferenceSettings, ModelClass,
    ModelIdentity, Point, SourceFrame as BatchSourceFrame, TimeUnit, TimestampStage,
    MAX_IDENTIFIER_CHARACTERS,
};
use crate::framing::write_framed_message;
use crate::geometry::{calibrate_detection_batch, AppliedCalibrationContext, GeometryError};
use crate::public_state::PublicStateConfig;
use crate::public_state_lifecycle::{
    CurrentSceneState, PublicStateDeliveryV1, SceneStateLifecycle, StateLifecycleConfig,
};
use crate::scene_transport::{LatestSceneWriter, SceneTransportConfig, SceneTransportStats};
use crate::tracking::{SceneTrackerV1, TrackingConfigV1};
use crate::{
    calibration::{TableCalibrationV1, ValidatedTableCalibrationV1},
    encode_scene_snapshot_v1, ImageSpace as CalibrationImageSpace, SceneSnapshotV1,
};

use crate::video_timeline::{parse_ffprobe_timeline, VideoTimeline, VideoTimelineLimits};

use super::{
    decode_yolo_v8_output, prepare_rgb8, prepare_rgb_image, Detection, LetterboxGeometry,
    PreparedInput,
};

const INPUT_NAME: &str = "images";
const OUTPUT_NAME: &str = "output0";
const EXPECTED_CLASS_COUNT: usize = 2;
const SUPPORTED_IMAGE_EXTENSIONS: [&str; 2] = ["jpg", "png"];
const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const PUBLIC_STATE_DELIVERY_SCHEMA_VERSION: u8 = 1;
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
    pub onnx_path: PathBuf,
    pub source_checkpoint_path: PathBuf,
    pub provider: ProviderSelection,
    pub image_size: u32,
    pub confidence: f32,
    pub iou: f32,
    pub max_detections: usize,
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
}

/// Reusable ONNX session for single-frame RGB inference.
///
/// The mutable session is deliberately not shared: a live runtime should keep
/// this value on its inference worker and hand it only one frame at a time.
/// The environment field is declared after the session so the session is
/// dropped before its ONNX Runtime environment.
pub struct FrameInferenceEngine {
    session: InferenceSession,
    _environment: Environment,
    input_shape: Vec<i64>,
    expected_output_shape: Vec<i64>,
    class_names: Vec<String>,
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
        validate_frame_inference_options(&options)?;
        let model_contract =
            load_and_verify_manifest(&options.onnx_path, &options.source_checkpoint_path)?;
        Self::from_verified_model(environment, options, model_contract).map(|(engine, _)| engine)
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
            &options.onnx_path,
            options.provider,
            options.intra_op_threads,
            None,
        )?;
        let (input_shape, actual_output_shape) =
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
                _environment: environment,
                input_shape,
                expected_output_shape,
                class_names: model_contract.class_names,
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

    /// Run preprocessing, ONNX inference, and YOLO decoding on one RGB frame.
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
        let prepared = prepare_rgb8(source_width, source_height, pixels, self.image_size as u32)
            .context("failed to preprocess RGB frame")?;
        let preprocess_ms = elapsed_ms(preprocess_start);

        let mut output = self.infer_prepared(&prepared)?;
        output.timings.preprocess_ms = preprocess_ms;
        output.timings.engine_total_ms = elapsed_ms(engine_start);
        Ok(output)
    }

    fn infer_prepared(&mut self, prepared: &PreparedInput) -> Result<FrameInferenceOutput> {
        let engine_start = Instant::now();

        let inference_start = Instant::now();
        let (output_shape, output_values) =
            run_session(&mut self.session, &prepared.values, self.image_size)
                .context("ONNX inference failed for RGB frame")?;
        let inference_ms = elapsed_ms(inference_start);
        ensure!(
            output_shape == self.expected_output_shape,
            "ONNX output tensor shape changed at runtime: {output_shape:?}"
        );

        let decode_start = Instant::now();
        let detections = decode_yolo_v8_output(
            &output_shape
                .iter()
                .map(|dimension| *dimension as usize)
                .collect::<Vec<_>>(),
            &output_values,
            &self.class_names,
            prepared.geometry,
            self.confidence,
            self.iou,
            self.max_detections,
        )
        .context("failed to decode ONNX output for RGB frame")?;
        let decode_and_nms_ms = elapsed_ms(decode_start);

        Ok(FrameInferenceOutput {
            source_width_px: prepared.geometry.source_width,
            source_height_px: prepared.geometry.source_height,
            detections,
            timings: FrameInferenceTimings {
                preprocess_ms: 0.0,
                inference_ms,
                decode_and_nms_ms,
                engine_total_ms: elapsed_ms(engine_start),
            },
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

    fn input_shape(&self) -> &[i64] {
        &self.input_shape
    }

    fn output_shape(&self) -> &[i64] {
        &self.expected_output_shape
    }
}

/// All runtime-affecting inference options are explicit in the audit result.
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
    pub retained_provider_profile: Option<PathBuf>,
    pub dump_preprocessed_directory: Option<PathBuf>,
    /// Source image coordinate space. This describes the input image bytes;
    /// letterbox resizing is later reversed before detector positions publish.
    pub source_image_space: ImageSpace,
    /// Optional file or loopback TCP sink; synchronous delivery time is reported separately.
    pub detection_batch_output: Option<BatchOutputOptions>,
    /// Optional diagnostic geometry pass. It requires detection-batch output
    /// so the calibrated measurements are reproducible from retained input.
    pub calibration: Option<CalibrationAuditOptions>,
    /// Optional paced scene-snapshot replay. It requires detection-batch
    /// output so tracks can always be reconstructed from raw observations.
    pub scene_snapshot_output: Option<SceneSnapshotOutputOptions>,
    /// Optional source presentation rate. When set, replay uses a one-frame
    /// latest-work mailbox so inference overload replaces stale pending input.
    pub replay_rate_fps: Option<f64>,
    /// Number of times the recorded corpus is presented during paced replay.
    /// Repeated corpus entries get fresh synthetic source frame indexes.
    pub replay_repeat_count: usize,
    /// Hard limit on scheduled paced frames. Required when replaying a corpus
    /// more than once so accidental long runs fail before inference starts.
    pub maximum_replay_frames: Option<usize>,
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
    /// Shared model, provider, tracking, and output settings. `input_path` is
    /// the video file; still-image replay fields must remain unset.
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
    pub maximum_detail_records: usize,
}

/// Explicit scene output and already validated tracking policy for replay.
#[derive(Debug, Clone)]
pub struct SceneSnapshotOutputOptions {
    pub destination: SceneSnapshotOutputDestination,
    pub tracking: TrackingConfigV1,
    /// Explicit schema selection for native current-state TCP delivery.
    /// JSONL remains raw scene-snapshot v1 and therefore must leave this unset.
    pub delivery_schema_version: Option<u8>,
}

/// Recorded-replay scene output. TCP export uses a bounded latest-only worker
/// so socket delays do not run on the inference thread.
#[derive(Debug, Clone)]
pub enum SceneSnapshotOutputDestination {
    JsonlFile {
        path: PathBuf,
    },
    LoopbackTcp {
        address: SocketAddr,
        maximum_message_bytes: usize,
        connect_timeout: Duration,
        write_timeout: Duration,
        reconnect_initial_delay: Duration,
        reconnect_maximum_delay: Duration,
    },
}

/// Explicit identities and destination for per-frame contract output.
#[derive(Debug, Clone)]
pub struct BatchOutputOptions {
    pub destination: BatchOutputDestination,
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

/// A replay-only sink. TCP uses a synchronous connection to a loopback receiver;
/// it is not the live pipeline's bounded asynchronous transport.
#[derive(Debug, Clone)]
pub enum BatchOutputDestination {
    JsonlFile {
        path: PathBuf,
    },
    LoopbackTcp {
        address: SocketAddr,
        maximum_message_bytes: usize,
        connect_timeout: Duration,
        write_timeout: Duration,
    },
}

/// Explicit inference backend policy for this recorded-image audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderSelection {
    /// Disable automatic accelerators and register the portable CPU provider.
    Cpu,
    /// Let ONNX Runtime choose its max-efficiency device policy.
    Auto,
    /// Explicitly request the native OpenVINO Runtime CPU backend.
    #[serde(rename = "openvino")]
    Openvino,
}

impl ProviderSelection {
    fn requested_provider_name(self) -> &'static str {
        match self {
            Self::Cpu => CPU::NAME,
            Self::Auto => "AutoDevicePolicy::MaxEfficiency",
            Self::Openvino => "OpenVINO CPU (native)",
        }
    }

    fn binding(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Openvino => (
                "openvino-rs",
                "0.11.1",
                "Native OpenVINO CPU backend selected and invoked; this API does not expose per-node fallback assignment, so node counts are unavailable",
            ),
            Self::Cpu | Self::Auto => (
                "ort",
                "2.0.0-rc.15",
                "ONNX Runtime per-node profile from a separate identically configured session; empty observations mean unavailable",
            ),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AuditReport {
    pub schema_version: u32,
    pub workload: &'static str,
    pub parity_scope: &'static str,
    pub model: ModelReport,
    pub runtime: RuntimeReport,
    pub preprocessing: PreprocessingReport,
    pub settings: SettingsReport,
    pub timings: RunTimings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detection_batch_output: Option<BatchOutputReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationAuditReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_snapshot_output: Option<SceneSnapshotOutputReport>,
    pub results: Vec<ImageResult>,
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
    pub artifacts: VideoReplayArtifactsReport,
    pub effective_tracking_configuration: Option<Value>,
    pub batch_engine_instance_id: Option<String>,
    pub source_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal_accuracy: Option<Value>,
    pub unsupported_measurements: Vec<&'static str>,
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
    pub artifact_writes_included_in_measurement: bool,
    pub offline_decode_excluded_from_measurement: bool,
}

#[derive(Debug, Serialize)]
pub struct SceneSnapshotOutputReport {
    pub destination: String,
    pub format: &'static str,
    pub mode: &'static str,
    pub snapshot_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connect_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_message_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connect_timeout_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_timeout_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<SceneTransportStats>,
    pub ball_class_id: u32,
    pub cue_class_id: Option<u32>,
    pub camera_profile_id: String,
    pub association_gate_px: f64,
    pub association_prediction_horizon_ms: f64,
    pub lost_after_ms: f64,
    pub maximum_track_age_ms: f64,
    pub maximum_active_tracks: usize,
    pub recent_observation_capacity: usize,
    pub configuration_revision: u64,
    pub rectification_revision: u64,
    pub mean_tracking_ms: f64,
    pub mean_encode_enqueue_ms: f64,
    pub source_time: &'static str,
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
    /// Null means the frame could not be projected; an empty array means a
    /// successful complete observation contained no detections.
    pub measurements: Option<Vec<CalibratedMeasurement>>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    Valid,
    Unavailable,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct CalibratedMeasurement {
    pub detection_index: u64,
    pub center_image_px: Point,
    pub center_table_mm: crate::calibration::TablePointMm,
}

#[derive(Debug, Serialize)]
pub struct BatchOutputReport {
    pub destination: String,
    pub format: &'static str,
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connect_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum_message_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connect_timeout_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_timeout_ms: Option<f64>,
    pub engine_instance_id: String,
    pub source_session_id: String,
    pub batch_count: usize,
    pub source_time: &'static str,
}

enum DetectionBatchSink {
    Jsonl(BufWriter<fs::File>),
    LoopbackTcp {
        stream: TcpStream,
        maximum_message_bytes: usize,
    },
}

#[derive(Debug, Serialize)]
pub struct ModelReport {
    pub onnx_path: String,
    pub onnx_sha256: String,
    pub source_checkpoint_path: String,
    pub source_checkpoint_sha256: String,
    pub manifest_path: String,
    pub input_name: String,
    pub input_shape: Vec<i64>,
    pub output_name: String,
    pub output_shape: Vec<i64>,
    pub class_names: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct RuntimeReport {
    pub binding: &'static str,
    pub binding_version: &'static str,
    pub requested_execution_provider: &'static str,
    pub observed_execution_providers: Vec<String>,
    pub provider_evidence: &'static str,
    pub provider_event_count: usize,
    pub provider_node_counts: BTreeMap<String, usize>,
    pub retained_profile_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PreprocessingReport {
    pub color_order: &'static str,
    pub layout: &'static str,
    pub dtype: &'static str,
    pub normalization: &'static str,
    pub resize: &'static str,
    pub padding: &'static str,
    pub parity_note: &'static str,
    pub optional_tensor_dump_format: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SettingsReport {
    pub provider: ProviderSelection,
    pub source_image_space: ImageSpace,
    pub image_size: u32,
    pub confidence: f32,
    pub iou: f32,
    pub max_detections: usize,
    pub intra_op_threads: Option<usize>,
    pub warmup_runs: usize,
    pub replay_rate_fps: Option<f64>,
    pub replay_repeat_count: usize,
    pub maximum_replay_frames: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct RunTimings {
    pub image_count: usize,
    pub provider_probe_session_init_ms: f64,
    pub provider_probe_inference_ms: f64,
    pub measurement_session_init_ms: f64,
    pub warmup_total_ms: f64,
    pub total_recorded_image_ms: f64,
    pub summed_image_pipeline_ms: f64,
    pub mean_image_decode_ms: f64,
    pub mean_preprocess_ms: f64,
    pub mean_inference_ms: f64,
    pub mean_decode_and_nms_ms: f64,
    pub mean_end_to_end_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_scene_tracking_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_scene_encode_enqueue_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_calibration_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_to_calibration_complete_ms: Option<LatencyPercentiles>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_export_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paced_replay: Option<PacedReplayReport>,
}

#[derive(Debug, Serialize)]
pub struct ImageResult {
    pub path: String,
    /// Zero-based scheduled source index. Repeated corpus entries receive new
    /// indexes, while gaps expose latest-frame drops.
    pub source_frame_index: usize,
    pub encoded_image_sha256: String,
    pub source_width: u32,
    pub source_height: u32,
    pub letterbox: LetterboxGeometry,
    pub timings: ImageTimings,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_schedule_lateness_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_to_inference_start_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_to_inference_complete_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_to_calibration_complete_ms: Option<f64>,
    pub preprocessed_tensor_path: Option<String>,
    pub detections: Vec<Detection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<FrameCalibrationReport>,
}

#[derive(Debug, Serialize)]
pub struct PacedReplayReport {
    pub requested_source_fps: f64,
    pub corpus_frame_count: usize,
    pub replay_repeat_count: usize,
    pub scheduled_frame_count: usize,
    pub decoded_frame_count: usize,
    pub processed_frame_count: usize,
    /// Fixed half-open source interval `[replay_start, replay_start + scheduled_frame_count / requested_source_fps)`.
    /// Both throughput rates below use this exact duration and count only
    /// events whose monotonic event time falls inside it.
    pub measurement_window_duration_ms: f64,
    /// Successfully decoded and offered inputs whose decode-complete event
    /// occurred inside the fixed measurement window.
    pub decoded_frame_count_in_measurement_window: usize,
    /// Successful inference outputs whose inference-complete event occurred
    /// inside the same fixed measurement window.
    pub inference_complete_frame_count_in_measurement_window: usize,
    /// Inputs decoded after the measurement window ended while the producer
    /// was finishing its scheduled work.
    pub decoded_frame_count_after_measurement_window: usize,
    /// Successful inference outputs completed after the measurement window,
    /// including work drained after the replay source finished.
    pub inference_complete_frame_count_after_measurement_window: usize,
    pub replaced_pending_frame_count: usize,
    pub source_schedule_deadline_misses: usize,
    pub mean_source_schedule_lateness_ms: f64,
    pub maximum_source_schedule_lateness_ms: f64,
    pub observed_source_fps: Option<f64>,
    pub processed_frame_fps: f64,
    pub source_to_inference_start_ms: LatencyPercentiles,
    pub source_to_inference_complete_ms: LatencyPercentiles,
}

#[derive(Debug, Serialize)]
pub struct LatencyPercentiles {
    pub sample_count: usize,
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ImageTimings {
    pub image_decode_ms: f64,
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub decode_and_nms_ms: f64,
    pub end_to_end_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_tracking_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_encode_enqueue_ms: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ExportManifest {
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
}

struct ProviderEvidence {
    observed_providers: Vec<String>,
    event_count: usize,
    node_counts: BTreeMap<String, usize>,
    retained_profile_path: Option<PathBuf>,
}

struct TimedInput {
    image: RgbImage,
    prepared: PreparedInput,
    decode_ms: f64,
    preprocess_ms: f64,
}

struct DecodedReplayFrame {
    source_index: usize,
    path: PathBuf,
    encoded_image_sha256: String,
    image: RgbImage,
    decode_ms: f64,
    scheduled_at: Instant,
    source_arrived_at: Instant,
}

#[derive(Default)]
struct PacedMailboxState {
    latest: Option<DecodedReplayFrame>,
    closed: bool,
    cancelled: bool,
    producer_error: Option<String>,
    decoded_frame_count: usize,
    replaced_pending_frame_count: usize,
    source_schedule_deadline_misses: usize,
    total_source_schedule_lateness_ms: f64,
    maximum_source_schedule_lateness_ms: f64,
    rate_counts: PacedReplayRateCounts,
}

#[derive(Clone, Default)]
struct PacedReplayMailbox {
    shared: Arc<(Mutex<PacedMailboxState>, Condvar)>,
    measurement_window: Option<PacedReplayMeasurementWindow>,
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

#[derive(Debug, Default, Clone, Copy)]
struct PacedReplayRateCounts {
    decoded_in_measurement_window: usize,
    decoded_after_measurement_window: usize,
    inference_complete_in_measurement_window: usize,
    inference_complete_after_measurement_window: usize,
}

impl PacedReplayRateCounts {
    fn record_decoded(&mut self, event_at: Instant, window: PacedReplayMeasurementWindow) {
        debug_assert!(event_at >= window.start);
        if window.contains(event_at) {
            self.decoded_in_measurement_window += 1;
        } else if event_at >= window.end {
            self.decoded_after_measurement_window += 1;
        }
    }

    fn record_inference_complete(
        &mut self,
        event_at: Instant,
        window: PacedReplayMeasurementWindow,
    ) {
        debug_assert!(event_at >= window.start);
        if window.contains(event_at) {
            self.inference_complete_in_measurement_window += 1;
        } else if event_at >= window.end {
            self.inference_complete_after_measurement_window += 1;
        }
    }
}

struct FrameToProcess {
    source_index: usize,
    path: PathBuf,
    encoded_image_sha256: String,
    timed_input: TimedInput,
    scheduled_at: Option<Instant>,
    source_arrived_at: Option<Instant>,
}

struct FrameProcessContext<'a> {
    inference_engine: &'a mut FrameInferenceEngine,
    class_names: &'a [String],
    options: &'a AuditOptions,
    onnx_sha256: &'a str,
    source_checkpoint_sha256: &'a str,
    batch_sink: &'a mut Option<DetectionBatchSink>,
    scene_output: &'a mut Option<SceneSnapshotOutputRuntime>,
    batch_clock_origin: Instant,
    preprocessed_directory: Option<&'a Path>,
}

struct SceneSnapshotOutputRuntime {
    destination: String,
    format: &'static str,
    connect_ms: Option<f64>,
    maximum_message_bytes: Option<usize>,
    connect_timeout_ms: Option<f64>,
    write_timeout_ms: Option<f64>,
    tracker: SceneTrackerV1,
    lifecycle: Option<SceneStateLifecycle>,
    sink: SceneSnapshotSink,
    snapshot_count: usize,
    total_tracking_ms: f64,
    total_encode_enqueue_ms: f64,
    transport_stats: Option<SceneTransportStats>,
}

enum SceneSnapshotSink {
    Jsonl(BufWriter<fs::File>),
    LoopbackTcp {
        writer: LatestSceneWriter,
        maximum_message_bytes: usize,
    },
}

impl SceneSnapshotOutputRuntime {
    fn finish(&mut self) {
        if let SceneSnapshotSink::LoopbackTcp { writer, .. } = &mut self.sink {
            self.transport_stats = Some(writer.shutdown());
        }
    }
}

/// Run unpaced inference over one image or a directory of PNG/JPEG images.
///
/// A short profiling session first records actual ONNX Runtime node-provider
/// events. Timed results then come from a second, non-profiled session so that
/// ORT trace collection does not contaminate the reported per-image timings.
pub fn run_audit(options: &AuditOptions) -> Result<AuditReport> {
    validate_options(options)?;
    let image_paths = collect_images(&options.input_path)?;
    let paced_scheduled_frame_count = options
        .replay_rate_fps
        .map(|_| {
            validate_replay_frame_budget(
                image_paths.len(),
                options.replay_repeat_count,
                options.maximum_replay_frames,
            )
        })
        .transpose()?;
    let manifest = load_and_verify_manifest(&options.onnx_path, &options.source_checkpoint_path)?;
    if let Some(scene_output) = &options.scene_snapshot_output {
        ensure!(
            (scene_output.tracking.ball_class_id() as usize) < manifest.class_names.len(),
            "scene tracking ball class ID is absent from the model manifest"
        );
        if let Some(cue_class_id) = scene_output.tracking.cue_class_id() {
            ensure!(
                (cue_class_id as usize) < manifest.class_names.len(),
                "scene tracking cue class ID is absent from the model manifest"
            );
        }
    }
    let onnx_sha256 = manifest.artifact_sha256.clone();
    let expected_output_shape = vec![
        1_i64,
        (4 + manifest.class_names.len()) as i64,
        expected_anchor_count(options.image_size)? as i64,
    ];

    let environment = ort::init()
        .with_name("billiards-perception-recorded-image-audit")
        .with_telemetry(false)
        .build()
        .context("failed to initialize ONNX Runtime")?;

    let first = load_and_prepare(&image_paths[0], options.image_size)?.prepared;
    let (profile_evidence, provider_probe_session_init_ms, provider_probe_inference_ms, first) =
        profile_provider(
            &environment,
            options,
            &manifest.class_names,
            &expected_output_shape,
            first,
        )?;

    let (mut inference_engine, measurement_session_init_ms) =
        FrameInferenceEngine::from_verified_model(
            environment.clone(),
            FrameInferenceOptions {
                onnx_path: options.onnx_path.clone(),
                source_checkpoint_path: options.source_checkpoint_path.clone(),
                provider: options.provider,
                image_size: options.image_size,
                confidence: options.confidence,
                iou: options.iou,
                max_detections: options.max_detections,
                intra_op_threads: options.intra_op_threads,
            },
            manifest.clone(),
        )?;
    ensure!(
        inference_engine.output_shape() == expected_output_shape,
        "measurement output signature changed after provider probe: {:?}",
        inference_engine.output_shape()
    );
    let input_shape = inference_engine.input_shape().to_vec();
    let output_shape = inference_engine.output_shape().to_vec();
    let warmup_start = Instant::now();
    for _ in 0..options.warmup_runs {
        let _ = inference_engine.infer_prepared(&first)?;
    }
    let warmup_total_ms = elapsed_ms(warmup_start);

    if let Some(directory) = &options.dump_preprocessed_directory {
        fs::create_dir_all(directory).with_context(|| {
            format!(
                "failed to create preprocessed tensor directory {}",
                directory.display()
            )
        })?;
    }
    // Hashes are provenance, not a measured inference stage. Calculate them
    // before the recorded-image timing window.
    let image_hashes = image_paths
        .iter()
        .map(|path| sha256_file(path))
        .collect::<Result<Vec<_>>>()?;

    let (mut batch_sink, batch_connect_ms) = match options.detection_batch_output.as_ref() {
        Some(batch_options) => {
            let (sink, connect_ms) = open_detection_batch_sink(batch_options)?;
            (Some(sink), connect_ms)
        }
        None => (None, None),
    };
    let mut scene_output = options
        .scene_snapshot_output
        .as_ref()
        .map(|scene_options| {
            let batch_options = options
                .detection_batch_output
                .as_ref()
                .context("scene snapshot output requires producer identity")?;
            open_scene_snapshot_output(scene_options, batch_options)
        })
        .transpose()?;
    let measured_start = Instant::now();
    let mut context = FrameProcessContext {
        inference_engine: &mut inference_engine,
        class_names: &manifest.class_names,
        options,
        onnx_sha256: &onnx_sha256,
        source_checkpoint_sha256: &manifest.source_checkpoint_sha256,
        batch_sink: &mut batch_sink,
        scene_output: &mut scene_output,
        batch_clock_origin: measured_start,
        preprocessed_directory: options.dump_preprocessed_directory.as_deref(),
    };
    let (results, batch_export_ms, paced_replay, total_recorded_image_ms) =
        if let Some(requested_fps) = options.replay_rate_fps {
            let scheduled_frame_count = paced_scheduled_frame_count
                .context("paced replay frame count was not validated")?;
            run_paced_replay(
                &image_paths,
                &image_hashes,
                requested_fps,
                options.replay_repeat_count,
                scheduled_frame_count,
                measured_start,
                &mut context,
            )?
        } else {
            let mut results = Vec::with_capacity(image_paths.len());
            let mut batch_export_ms = 0.0;
            for (index, (path, image_hash)) in image_paths.iter().zip(image_hashes).enumerate() {
                let timed_input = load_and_prepare(path, options.image_size)?;
                let (result, export_ms, _) = process_frame(
                    &mut context,
                    FrameToProcess {
                        source_index: index,
                        path: path.clone(),
                        encoded_image_sha256: image_hash,
                        timed_input,
                        scheduled_at: None,
                        source_arrived_at: None,
                    },
                    index + 1,
                )?;
                results.push(result);
                batch_export_ms += export_ms;
            }
            (results, batch_export_ms, None, elapsed_ms(measured_start))
        };

    if let Some(scene_runtime) = scene_output.as_mut() {
        scene_runtime.finish();
    }
    let run_timings = summarize_timings(
        &results,
        provider_probe_session_init_ms,
        provider_probe_inference_ms,
        measurement_session_init_ms,
        warmup_total_ms,
        total_recorded_image_ms,
        batch_sink.is_some().then_some(batch_export_ms),
        paced_replay,
    );

    let detection_batch_output = options
        .detection_batch_output
        .as_ref()
        .map(|batch_options| BatchOutputReport {
            destination: batch_destination_label(&batch_options.destination),
            format: batch_output_format(&batch_options.destination),
            mode: batch_output_mode(
                &batch_options.destination,
                options.replay_rate_fps.is_some(),
            ),
            connect_ms: batch_connect_ms,
            maximum_message_bytes: batch_maximum_message_bytes(&batch_options.destination),
            connect_timeout_ms: batch_connect_timeout_ms(&batch_options.destination),
            write_timeout_ms: batch_write_timeout_ms(&batch_options.destination),
            engine_instance_id: batch_options.engine_instance_id.clone(),
            source_session_id: batch_options.source_session_id.clone(),
            batch_count: results.len(),
            source_time: if options.replay_rate_fps.is_some() {
                "synthetic_monotonic_frame_arrival"
            } else {
                "unavailable_for_still_image_replay"
            },
        });
    let calibration = options.calibration.as_ref().map(|calibration_options| {
        summarize_calibration(calibration_options, options.source_image_space, &results)
    });
    let scene_snapshot_output = options
        .scene_snapshot_output
        .as_ref()
        .zip(scene_output.as_ref())
        .map(|(output_options, runtime)| summarize_scene_snapshot_output(output_options, runtime));

    Ok(AuditReport {
        schema_version: 1,
        workload: if options.replay_rate_fps.is_some() {
            "recorded_images_paced_latest_frame"
        } else {
            "recorded_static_images_unpaced"
        },
        parity_scope: "1280-square Ultralytics equivalence harness; the configured CoreML package declares a fixed 1280-square input, but package lineage and CoreML-versus-ONNX parity remain unverified",
        model: ModelReport {
            onnx_path: options.onnx_path.display().to_string(),
            onnx_sha256,
            source_checkpoint_path: options.source_checkpoint_path.display().to_string(),
            source_checkpoint_sha256: manifest.source_checkpoint_sha256,
            manifest_path: manifest.manifest_path.display().to_string(),
            input_name: INPUT_NAME.to_owned(),
            input_shape,
            output_name: OUTPUT_NAME.to_owned(),
            output_shape,
            class_names: manifest.class_names,
        },
        runtime: RuntimeReport {
            binding: options.provider.binding().0,
            binding_version: options.provider.binding().1,
            requested_execution_provider: options.provider.requested_provider_name(),
            observed_execution_providers: profile_evidence.observed_providers,
            provider_evidence: options.provider.binding().2,
            provider_event_count: profile_evidence.event_count,
            provider_node_counts: profile_evidence.node_counts,
            retained_profile_path: profile_evidence.retained_profile_path.map(|path| path.display().to_string()),
        },
        preprocessing: PreprocessingReport {
            color_order: "RGB",
            layout: "NCHW",
            dtype: "float32",
            normalization: "[0, 1]",
            resize: "aspect-preserving OpenCV INTER_LINEAR half-pixel sampling approximation",
            padding: "centered RGB=(114,114,114), fixed square, no stride auto-padding",
            parity_note: "PNG/JPEG are decoded by image-rs; this path is an OpenCV-compatible approximation until tensor comparison against the repository still-image fixtures is recorded, and JPEG decoder differences may remain",
            optional_tensor_dump_format: "raw little-endian float32 NCHW [1,3,image_size,image_size]",
        },
        settings: SettingsReport {
            provider: options.provider,
            source_image_space: options.source_image_space,
            image_size: options.image_size,
            confidence: options.confidence,
            iou: options.iou,
            max_detections: options.max_detections,
            intra_op_threads: options.intra_op_threads,
            warmup_runs: options.warmup_runs,
            replay_rate_fps: options.replay_rate_fps,
            replay_repeat_count: options.replay_repeat_count,
            maximum_replay_frames: options.maximum_replay_frames,
        },
        timings: run_timings,
        detection_batch_output,
        calibration,
        scene_snapshot_output,
        results,
    })
}

/// Run the current Rust detector against one selected video stream. Video is
/// decoded once into a bounded temporary raw-RGB spool before the measured
/// interval. This keeps ffmpeg startup and offline decode separate from the
/// inference schedule, while preserving source PTS for the realtime mode.
/// The temporary file is removed automatically on every return path.
pub fn run_video_audit(options: &VideoReplayOptions) -> Result<VideoReplayReport> {
    let audit = &options.audit;
    validate_video_audit_options(audit)?;
    ensure!(
        audit.replay_rate_fps.is_none() && audit.replay_repeat_count == 1,
        "video replay uses source PTS and does not accept still-image replay rate or repeat options"
    );
    ensure!(
        audit.calibration.is_none(),
        "recorded-video calibration output is not supported until source-clock semantics are defined for both replay modes"
    );
    ensure!(
        audit.dump_preprocessed_directory.is_none(),
        "video replay does not support per-frame tensor dumps"
    );
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
        options.details_jsonl_path.is_some() || options.maximum_detail_records == 0,
        "a nonzero detail-record limit requires a detail JSONL destination"
    );
    ensure!(
        matches!(
            audit
                .detection_batch_output
                .as_ref()
                .map(|output| &output.destination),
            None | Some(BatchOutputDestination::JsonlFile { .. })
        ),
        "recorded-video audits support file-based detection-batch output only"
    );
    ensure!(
        matches!(
            audit
                .scene_snapshot_output
                .as_ref()
                .map(|output| &output.destination),
            None | Some(SceneSnapshotOutputDestination::JsonlFile { .. })
        ),
        "recorded-video audits support file-based scene-snapshot output only"
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
                onnx_path: audit.onnx_path.clone(),
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

    let (mut batch_sink, _batch_connect_ms) = match audit.detection_batch_output.as_ref() {
        Some(batch_options) => {
            let (sink, connect_ms) = open_detection_batch_sink(batch_options)?;
            (Some(sink), connect_ms)
        }
        None => (None, None),
    };
    let mut scene_output = audit
        .scene_snapshot_output
        .as_ref()
        .map(|scene_options| {
            let batch_options = audit
                .detection_batch_output
                .as_ref()
                .context("scene output requires detection-batch producer identity")?;
            open_scene_snapshot_output(scene_options, batch_options)
        })
        .transpose()?;
    let mut detail_writer = options
        .details_jsonl_path
        .as_ref()
        .map(|path| create_new_buffered_file(path, "video detail trace"))
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
        onnx_sha256: &manifest.artifact_sha256,
        source_checkpoint_sha256: &manifest.source_checkpoint_sha256,
        batch_sink: &mut batch_sink,
        scene_output: &mut scene_output,
        batch_clock_origin: measurement_started,
        preprocessed_directory: None,
    };
    let mut next_source_index = 0usize;
    let mut processed_source_frames = 0usize;
    let mut latest_frame_replacements = 0usize;
    let mut processed_source_deadline_misses = 0usize;
    let mut completions_in_source_window = 0usize;
    let mut completions_after_source_window = 0usize;
    let mut source_to_start_samples = Vec::with_capacity(timeline.frame_count().min(4096));
    let mut source_to_complete_samples = Vec::with_capacity(timeline.frame_count().min(4096));
    let mut preprocess_total_ms = 0.0;
    let mut inference_total_ms = 0.0;
    let mut decode_and_nms_total_ms = 0.0;
    let mut batch_export_total_ms = 0.0;
    let mut trace_written = 0usize;
    let mut trace_limited = 0usize;
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
        let source_arrived_at = source_scheduled_at.or_else(|| Some(Instant::now()));
        let timed_input = prepare_decoded_image(image, 0.0, audit.image_size)?;
        let (result, export_ms, inference_completed_at) = process_frame(
            &mut context,
            FrameToProcess {
                source_index,
                path: video_path.clone(),
                encoded_image_sha256: String::new(),
                timed_input,
                scheduled_at: source_scheduled_at,
                source_arrived_at,
            },
            processed_source_frames + 1,
        )?;
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
    if let Some(writer) = detail_writer.as_mut() {
        writer
            .flush()
            .context("failed to flush video detail trace")?;
    }
    if let Some(scene_runtime) = scene_output.as_mut() {
        scene_runtime.finish();
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
            intra_op_threads: audit.intra_op_threads,
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
        },
        artifacts: VideoReplayArtifactsReport {
            detection_batch_path: audit
                .detection_batch_output
                .as_ref()
                .map(|output| batch_destination_label(&output.destination)),
            detection_batch_count: batch_sink
                .as_ref()
                .map(|_| processed_source_frames)
                .unwrap_or(0),
            scene_snapshot_path: audit.scene_snapshot_output.as_ref().map(|output| {
                match &output.destination {
                    SceneSnapshotOutputDestination::JsonlFile { path } => {
                        path.display().to_string()
                    }
                    SceneSnapshotOutputDestination::LoopbackTcp { .. } => unreachable!(),
                }
            }),
            scene_snapshot_count,
            detailed_trace_path: options
                .details_jsonl_path
                .as_ref()
                .map(|path| path.display().to_string()),
            detailed_trace_written: trace_written,
            detailed_trace_dropped_by_limit: trace_limited,
            artifact_writes_included_in_measurement: audit.detection_batch_output.is_some()
                || audit.scene_snapshot_output.is_some()
                || options.details_jsonl_path.is_some(),
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
        batch_engine_instance_id: audit
            .detection_batch_output
            .as_ref()
            .map(|output| output.engine_instance_id.clone()),
        source_session_id: audit
            .detection_batch_output
            .as_ref()
            .map(|output| output.source_session_id.clone()),
        temporal_accuracy: None,
        unsupported_measurements: vec![
            "camera_exposure_to_decode_latency",
            "browser_render_and_physical_display_latency",
            "process_cpu_utilization_and_memory_trend",
            "accelerator_per_node_assignment_for_native_openvino",
        ],
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
    timings: &'a ImageTimings,
    detections: &'a [Detection],
}

fn write_video_detail<W: Write>(
    writer: &mut W,
    source_frame_index: usize,
    source_pts_seconds: f64,
    result: &ImageResult,
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
) -> Result<(ImageResult, f64, Instant)> {
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
    let source_schedule_lateness_ms = frame.scheduled_at.map(|scheduled_at| {
        frame
            .source_arrived_at
            .unwrap_or(scheduled_at)
            .saturating_duration_since(scheduled_at)
            .as_secs_f64()
            * 1000.0
    });
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
    let mut calibration_report = None;
    let mut scene_tracking_ms = None;
    let mut scene_encode_enqueue_ms = None;
    let mut calibrated_for_scene = None;
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
            if context.scene_output.is_some() {
                calibrated_for_scene = Some((
                    calibration_options.applied_calibration_revision,
                    geometry_result,
                ));
            }
            calibration_ms = Some(elapsed_ms(calibration_start));
            source_to_calibration_complete_ms = frame.source_arrived_at.map(|source_arrived_at| {
                Instant::now()
                    .saturating_duration_since(source_arrived_at)
                    .as_secs_f64()
                    * 1000.0
            });
        }

        if let Some(scene_runtime) = context.scene_output.as_mut() {
            let published_at = scene_publication_timestamp(
                context.batch_clock_origin,
                &output_options.engine_instance_id,
            )?;
            let tracking_start = Instant::now();
            let mut snapshot = scene_runtime
                .tracker
                .process_with_precalibrated_batch(&batch, published_at, calibrated_for_scene)
                .context("failed to produce tracked scene snapshot")?;
            scene_tracking_ms = Some(elapsed_ms(tracking_start));

            // Tracking requires a valid timestamp to build its transactional
            // snapshot. Replace that provisional value with the publisher's
            // actual monotonic time immediately before serialization.
            snapshot.published_at = scene_publication_timestamp(
                context.batch_clock_origin,
                &output_options.engine_instance_id,
            )?;
            let current_state = if let Some(lifecycle) = scene_runtime.lifecycle.as_mut() {
                let source_arrived_at = frame
                    .source_arrived_at
                    .context("paced scene output is missing replay-local arrival time")?;
                ingest_replay_scene(
                    lifecycle,
                    &snapshot,
                    context.batch_clock_origin,
                    source_arrived_at,
                    Instant::now(),
                )?
            } else {
                None
            };
            let export_start = Instant::now();
            write_scene_snapshot(&mut scene_runtime.sink, &snapshot, current_state.as_ref())?;
            scene_encode_enqueue_ms = Some(elapsed_ms(export_start));
            scene_runtime.snapshot_count += 1;
            scene_runtime.total_tracking_ms += scene_tracking_ms.unwrap_or_default();
            scene_runtime.total_encode_enqueue_ms += scene_encode_enqueue_ms.unwrap_or_default();
        }
    }

    let preprocessed_tensor_path = context
        .preprocessed_directory
        .map(|directory| directory.join(format!("input-{:05}.f32le", frame.source_index)));
    if let Some(tensor_path) = &preprocessed_tensor_path {
        write_nchw_float32(tensor_path, &frame.timed_input.prepared.values)?;
    }

    Ok((
        ImageResult {
            path: frame.path.display().to_string(),
            source_frame_index: frame.source_index,
            encoded_image_sha256: frame.encoded_image_sha256,
            source_width: frame.timed_input.image.width(),
            source_height: frame.timed_input.image.height(),
            letterbox: frame.timed_input.prepared.geometry,
            timings: ImageTimings {
                image_decode_ms: frame.timed_input.decode_ms,
                preprocess_ms: frame.timed_input.preprocess_ms,
                inference_ms,
                decode_and_nms_ms,
                end_to_end_ms,
                calibration_ms,
                scene_tracking_ms,
                scene_encode_enqueue_ms,
            },
            source_schedule_lateness_ms,
            source_to_inference_start_ms,
            source_to_inference_complete_ms,
            source_to_calibration_complete_ms,
            preprocessed_tensor_path: preprocessed_tensor_path
                .map(|path| path.display().to_string()),
            detections,
            calibration: calibration_report,
        },
        batch_export_ms,
        inference_completed,
    ))
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
            measurements: Some(
                calibrated
                    .detections
                    .iter()
                    .map(|detection| CalibratedMeasurement {
                        detection_index: detection.detection_index,
                        center_image_px: detection.center_image_px,
                        center_table_mm: detection.center_table_mm,
                    })
                    .collect(),
            ),
        }),
        Err(error @ GeometryError::IncompatibleFrameCalibration(_))
        | Err(error @ GeometryError::DetectionProjection { .. }) => Ok(FrameCalibrationReport {
            status: CalibrationStatus::Unavailable,
            reason: Some(error.to_string()),
            measurements: None,
        }),
        Err(error) => {
            bail!("audit-created detection batch violated geometry input contract: {error}")
        }
    }
}

fn run_paced_replay(
    image_paths: &[PathBuf],
    image_hashes: &[String],
    requested_fps: f64,
    replay_repeat_count: usize,
    scheduled_frame_count: usize,
    replay_started: Instant,
    context: &mut FrameProcessContext<'_>,
) -> Result<(Vec<ImageResult>, f64, Option<PacedReplayReport>, f64)> {
    ensure!(
        image_paths.len() == image_hashes.len(),
        "paced replay image/hash counts differ"
    );
    let frame_period = replay_frame_period(requested_fps)?;
    let window_frame_count = u32::try_from(scheduled_frame_count)
        .context("paced replay measurement window frame count exceeds u32::MAX")?;
    let measurement_window_duration = frame_period
        .checked_mul(window_frame_count)
        .context("paced replay measurement window exceeds the monotonic clock range")?;
    let measurement_window =
        PacedReplayMeasurementWindow::new(replay_started, measurement_window_duration)?;

    let mailbox = PacedReplayMailbox::with_measurement_window(measurement_window);
    let producer_mailbox = mailbox.clone();
    let producer_paths = image_paths.to_vec();
    let producer_hashes = image_hashes.to_vec();
    let producer = thread::Builder::new()
        .name("paced-image-replay-source".to_owned())
        .spawn(move || {
            let result = produce_paced_frames(
                &producer_mailbox,
                &producer_paths,
                &producer_hashes,
                replay_repeat_count,
                scheduled_frame_count,
                replay_started,
                frame_period,
            );
            producer_mailbox.finish(result.as_ref().err().map(|error| format!("{error:#}")));
            result.map_err(|error| format!("{error:#}"))
        })
        .context("failed to start paced replay source thread")?;

    // A configured replay may span hours. Grow report storage only as frames
    // are actually processed rather than reserving the entire scheduled run.
    let mut results = Vec::with_capacity(scheduled_frame_count.min(1024));
    let mut batch_export_ms = 0.0;
    let mut source_to_inference_start_samples_ms =
        Vec::with_capacity(scheduled_frame_count.min(1024));
    let mut source_to_inference_complete_samples_ms =
        Vec::with_capacity(scheduled_frame_count.min(1024));
    let processing_result = (|| -> Result<()> {
        loop {
            let Some(frame) = mailbox.take_latest()? else {
                break;
            };
            let source_index = frame.source_index;
            let timed_input =
                prepare_decoded_image(frame.image, frame.decode_ms, context.options.image_size)?;
            let (result, export_ms, inference_completed_at) = process_frame(
                context,
                FrameToProcess {
                    source_index,
                    path: frame.path,
                    encoded_image_sha256: frame.encoded_image_sha256,
                    timed_input,
                    scheduled_at: Some(frame.scheduled_at),
                    source_arrived_at: Some(frame.source_arrived_at),
                },
                results.len() + 1,
            )?;
            mailbox.record_inference_complete(inference_completed_at)?;
            if let Some(latency) = result.source_to_inference_start_ms {
                source_to_inference_start_samples_ms.push(latency);
            }
            if let Some(latency) = result.source_to_inference_complete_ms {
                source_to_inference_complete_samples_ms.push(latency);
            }
            results.push(result);
            batch_export_ms += export_ms;
        }
        Ok(())
    })();
    if processing_result.is_err() {
        mailbox.cancel();
    }
    let producer_result = producer
        .join()
        .map_err(|_| anyhow::anyhow!("paced replay source thread panicked"))?;
    processing_result?;
    producer_result.map_err(anyhow::Error::msg)?;

    let total_recorded_image_ms = elapsed_ms(replay_started);
    let state = mailbox.snapshot()?;
    ensure!(
        state.rate_counts.decoded_in_measurement_window
            + state.rate_counts.decoded_after_measurement_window
            == state.decoded_frame_count,
        "paced replay decoded-frame measurement accounting is incomplete"
    );
    ensure!(
        state.rate_counts.inference_complete_in_measurement_window
            + state
                .rate_counts
                .inference_complete_after_measurement_window
            == results.len(),
        "paced replay inference-completion measurement accounting is incomplete"
    );
    let measurement_window_seconds = measurement_window.duration().as_secs_f64();
    let observed_source_fps = Some(frames_per_second(
        state.rate_counts.decoded_in_measurement_window,
        measurement_window,
    ));
    let processed_frame_fps = frames_per_second(
        state.rate_counts.inference_complete_in_measurement_window,
        measurement_window,
    );
    let report = PacedReplayReport {
        requested_source_fps: requested_fps,
        corpus_frame_count: image_paths.len(),
        replay_repeat_count,
        scheduled_frame_count,
        decoded_frame_count: state.decoded_frame_count,
        processed_frame_count: results.len(),
        measurement_window_duration_ms: measurement_window_seconds * 1000.0,
        decoded_frame_count_in_measurement_window: state.rate_counts.decoded_in_measurement_window,
        inference_complete_frame_count_in_measurement_window: state
            .rate_counts
            .inference_complete_in_measurement_window,
        decoded_frame_count_after_measurement_window: state
            .rate_counts
            .decoded_after_measurement_window,
        inference_complete_frame_count_after_measurement_window: state
            .rate_counts
            .inference_complete_after_measurement_window,
        replaced_pending_frame_count: state.replaced_pending_frame_count,
        source_schedule_deadline_misses: state.source_schedule_deadline_misses,
        mean_source_schedule_lateness_ms: if state.decoded_frame_count > 0 {
            state.total_source_schedule_lateness_ms / state.decoded_frame_count as f64
        } else {
            0.0
        },
        maximum_source_schedule_lateness_ms: state.maximum_source_schedule_lateness_ms,
        observed_source_fps,
        processed_frame_fps,
        source_to_inference_start_ms: percentile_summary(source_to_inference_start_samples_ms),
        source_to_inference_complete_ms: percentile_summary(
            source_to_inference_complete_samples_ms,
        ),
    };
    Ok((
        results,
        batch_export_ms,
        Some(report),
        total_recorded_image_ms,
    ))
}

fn produce_paced_frames(
    mailbox: &PacedReplayMailbox,
    image_paths: &[PathBuf],
    image_hashes: &[String],
    replay_repeat_count: usize,
    scheduled_frame_count: usize,
    replay_started: Instant,
    frame_period: Duration,
) -> Result<()> {
    ensure!(
        image_paths.len() == image_hashes.len(),
        "paced replay image/hash counts differ"
    );
    ensure!(
        !image_paths.is_empty(),
        "paced replay input corpus is empty"
    );
    ensure!(
        image_paths.len().checked_mul(replay_repeat_count) == Some(scheduled_frame_count),
        "paced replay scheduled frame count does not match its corpus and repeat count"
    );
    for source_index in 0..scheduled_frame_count {
        let corpus_index = source_index % image_paths.len();
        let path = &image_paths[corpus_index];
        let image_hash = &image_hashes[corpus_index];
        let frame_index = u32::try_from(source_index)
            .context("paced replay supports at most u32::MAX input frame indexes")?;
        let schedule_offset = frame_period
            .checked_mul(frame_index)
            .context("paced replay schedule exceeds the monotonic clock range")?;
        let scheduled_at = replay_started
            .checked_add(schedule_offset)
            .context("paced replay schedule exceeds the monotonic clock range")?;
        if !mailbox.wait_until(scheduled_at)? {
            break;
        }
        let (image, decode_ms) = load_rgb_image(path)?;
        let source_arrived_at = Instant::now();
        mailbox.offer(
            DecodedReplayFrame {
                source_index,
                path: path.clone(),
                encoded_image_sha256: image_hash.clone(),
                image,
                decode_ms,
                scheduled_at,
                source_arrived_at,
            },
            frame_period,
        )?;
    }
    Ok(())
}

impl PacedReplayMailbox {
    fn with_measurement_window(measurement_window: PacedReplayMeasurementWindow) -> Self {
        Self {
            shared: Arc::default(),
            measurement_window: Some(measurement_window),
        }
    }

    fn wait_until(&self, target: Instant) -> Result<bool> {
        let (lock, changed) = &*self.shared;
        let mut state = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("paced replay mailbox lock was poisoned"))?;
        loop {
            if state.cancelled {
                return Ok(false);
            }
            let remaining = target.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(true);
            }
            let (new_state, _) = changed
                .wait_timeout(state, remaining)
                .map_err(|_| anyhow::anyhow!("paced replay mailbox lock was poisoned"))?;
            state = new_state;
        }
    }

    fn offer(&self, frame: DecodedReplayFrame, frame_period: Duration) -> Result<()> {
        let (lock, changed) = &*self.shared;
        let mut state = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("paced replay mailbox lock was poisoned"))?;
        if state.cancelled {
            return Ok(());
        }
        let lateness = frame
            .source_arrived_at
            .saturating_duration_since(frame.scheduled_at);
        let lateness_ms = lateness.as_secs_f64() * 1000.0;
        state.decoded_frame_count += 1;
        if let Some(window) = self.measurement_window {
            state
                .rate_counts
                .record_decoded(frame.source_arrived_at, window);
        }
        state.total_source_schedule_lateness_ms += lateness_ms;
        state.maximum_source_schedule_lateness_ms =
            state.maximum_source_schedule_lateness_ms.max(lateness_ms);
        if lateness > frame_period {
            state.source_schedule_deadline_misses += 1;
        }
        if state.latest.replace(frame).is_some() {
            state.replaced_pending_frame_count += 1;
        }
        changed.notify_one();
        Ok(())
    }

    fn record_inference_complete(&self, event_at: Instant) -> Result<()> {
        if let Some(window) = self.measurement_window {
            let (lock, _) = &*self.shared;
            let mut state = lock
                .lock()
                .map_err(|_| anyhow::anyhow!("paced replay mailbox lock was poisoned"))?;
            state
                .rate_counts
                .record_inference_complete(event_at, window);
        }
        Ok(())
    }

    fn take_latest(&self) -> Result<Option<DecodedReplayFrame>> {
        let (lock, changed) = &*self.shared;
        let mut state = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("paced replay mailbox lock was poisoned"))?;
        loop {
            if let Some(frame) = state.latest.take() {
                return Ok(Some(frame));
            }
            if state.closed {
                if let Some(error) = &state.producer_error {
                    bail!("paced replay source failed: {error}");
                }
                return Ok(None);
            }
            state = changed
                .wait(state)
                .map_err(|_| anyhow::anyhow!("paced replay mailbox lock was poisoned"))?;
        }
    }

    fn finish(&self, producer_error: Option<String>) {
        let (lock, changed) = &*self.shared;
        if let Ok(mut state) = lock.lock() {
            state.producer_error = producer_error;
            state.closed = true;
            changed.notify_all();
        }
    }

    fn cancel(&self) {
        let (lock, changed) = &*self.shared;
        if let Ok(mut state) = lock.lock() {
            state.cancelled = true;
            changed.notify_all();
        }
    }

    fn snapshot(&self) -> Result<PacedMailboxStateSnapshot> {
        let (lock, _) = &*self.shared;
        let state = lock
            .lock()
            .map_err(|_| anyhow::anyhow!("paced replay mailbox lock was poisoned"))?;
        Ok(PacedMailboxStateSnapshot {
            decoded_frame_count: state.decoded_frame_count,
            replaced_pending_frame_count: state.replaced_pending_frame_count,
            source_schedule_deadline_misses: state.source_schedule_deadline_misses,
            total_source_schedule_lateness_ms: state.total_source_schedule_lateness_ms,
            maximum_source_schedule_lateness_ms: state.maximum_source_schedule_lateness_ms,
            rate_counts: state.rate_counts,
        })
    }
}

struct PacedMailboxStateSnapshot {
    decoded_frame_count: usize,
    replaced_pending_frame_count: usize,
    source_schedule_deadline_misses: usize,
    total_source_schedule_lateness_ms: f64,
    maximum_source_schedule_lateness_ms: f64,
    rate_counts: PacedReplayRateCounts,
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

fn replay_frame_period(requested_fps: f64) -> Result<Duration> {
    ensure!(
        requested_fps.is_finite() && requested_fps > 0.0,
        "paced replay rate must be finite and positive"
    );
    let period = Duration::try_from_secs_f64(1.0 / requested_fps)
        .context("paced replay rate cannot be represented as a frame period")?;
    ensure!(
        !period.is_zero(),
        "paced replay rate exceeds monotonic clock resolution"
    );
    Ok(period)
}

/// Resolve and validate the finite schedule for one paced recorded replay.
/// The source-frame wire contract uses a u32 index, and repeated corpora need
/// an explicit cap so short input recordings cannot accidentally run forever.
fn validate_replay_frame_budget(
    corpus_frame_count: usize,
    replay_repeat_count: usize,
    maximum_replay_frames: Option<usize>,
) -> Result<usize> {
    ensure!(corpus_frame_count > 0, "paced replay input corpus is empty");
    ensure!(
        replay_repeat_count > 0,
        "replay repeat count must be positive"
    );
    let scheduled_frame_count = corpus_frame_count
        .checked_mul(replay_repeat_count)
        .context("paced replay scheduled frame count overflows this platform")?;
    ensure!(
        scheduled_frame_count <= u32::MAX as usize,
        "paced replay supports at most u32::MAX scheduled frames"
    );
    if replay_repeat_count > 1 {
        ensure!(
            maximum_replay_frames.is_some(),
            "repeated paced replay requires --maximum-replay-frames"
        );
    }
    if let Some(maximum_replay_frames) = maximum_replay_frames {
        ensure!(
            maximum_replay_frames > 0,
            "maximum replay frame count must be positive"
        );
        ensure!(
            scheduled_frame_count <= maximum_replay_frames,
            "paced replay schedules {scheduled_frame_count} frames, exceeding the configured maximum of {maximum_replay_frames}"
        );
    }
    Ok(scheduled_frame_count)
}

fn validate_options(options: &AuditOptions) -> Result<()> {
    validate_audit_options(options, false)
}

fn validate_video_audit_options(options: &AuditOptions) -> Result<()> {
    validate_audit_options(options, true)
}

fn validate_audit_options(
    options: &AuditOptions,
    video_source_timestamps_available: bool,
) -> Result<()> {
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
    ensure!(
        options.replay_repeat_count > 0,
        "replay repeat count must be positive"
    );
    if let Some(maximum_replay_frames) = options.maximum_replay_frames {
        ensure!(
            maximum_replay_frames > 0,
            "maximum replay frame count must be positive"
        );
    }
    if let Some(replay_rate_fps) = options.replay_rate_fps {
        replay_frame_period(replay_rate_fps)?;
    } else {
        ensure!(
            options.replay_repeat_count == 1,
            "repeated recorded input requires --replay-rate-fps"
        );
    }
    if let Some(batch_output) = &options.detection_batch_output {
        validate_batch_identifier(&batch_output.engine_instance_id, "engine instance ID")?;
        validate_batch_identifier(&batch_output.source_session_id, "source session ID")?;
        ensure!(
            options.max_detections <= u32::MAX as usize,
            "maximum detections exceeds the batch contract limit"
        );
        match &batch_output.destination {
            BatchOutputDestination::JsonlFile { path } => ensure!(
                !path.exists(),
                "detection batch output already exists: {}",
                path.display()
            ),
            BatchOutputDestination::LoopbackTcp {
                address,
                maximum_message_bytes,
                connect_timeout,
                write_timeout,
            } => {
                ensure!(
                    address.ip().is_loopback(),
                    "batch TCP address must be loopback"
                );
                ensure!(
                    *maximum_message_bytes > 0 && *maximum_message_bytes <= u32::MAX as usize,
                    "maximum framed message size must fit a non-zero u32"
                );
                ensure!(
                    !connect_timeout.is_zero() && !write_timeout.is_zero(),
                    "batch TCP connect and write timeouts must be positive"
                );
            }
        }
        if video_source_timestamps_available {
            ensure!(
                matches!(
                    &batch_output.destination,
                    BatchOutputDestination::JsonlFile { .. }
                ),
                "recorded-video audits support file-based detection-batch output only"
            );
        }
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
        ensure!(
            options.replay_rate_fps.is_some() || video_source_timestamps_available,
            "scene snapshot output requires paced replay so source timestamps are available"
        );
        match &scene_output.destination {
            SceneSnapshotOutputDestination::JsonlFile { path } => {
                ensure!(
                    scene_output.delivery_schema_version.is_none(),
                    "native public-state delivery schema selection is supported only for TCP output"
                );
                ensure!(
                    !path.exists(),
                    "scene snapshot output already exists: {}",
                    path.display()
                );
                if let Some(BatchOutputOptions {
                    destination: BatchOutputDestination::JsonlFile { path: batch_path },
                    ..
                }) = &options.detection_batch_output
                {
                    ensure!(
                        batch_path != path,
                        "detection batch and scene snapshot output must use different paths"
                    );
                }
            }
            SceneSnapshotOutputDestination::LoopbackTcp {
                address,
                maximum_message_bytes,
                connect_timeout,
                write_timeout,
                reconnect_initial_delay,
                reconnect_maximum_delay,
            } => {
                ensure!(
                    !video_source_timestamps_available,
                    "recorded-video audits support file-based scene-snapshot output only"
                );
                ensure!(
                    scene_output.delivery_schema_version
                        == Some(PUBLIC_STATE_DELIVERY_SCHEMA_VERSION),
                    "scene TCP output requires the supported public-state delivery schema version 1"
                );
                ensure!(
                    address.ip().is_loopback(),
                    "scene TCP address must be loopback"
                );
                ensure!(
                    *maximum_message_bytes > 0 && *maximum_message_bytes <= u32::MAX as usize,
                    "scene maximum framed message size must fit a non-zero u32"
                );
                ensure!(
                    !connect_timeout.is_zero() && !write_timeout.is_zero(),
                    "scene TCP connect and write timeouts must be positive"
                );
                ensure!(
                    !reconnect_initial_delay.is_zero()
                        && reconnect_initial_delay <= reconnect_maximum_delay,
                    "scene TCP reconnect delays must be positive and initial cannot exceed maximum"
                );
            }
        }
        if let Some(calibration) = &options.calibration {
            validate_scene_calibration_context(&scene_output.tracking, calibration)?;
        }
    }
    Ok(())
}

fn validate_frame_inference_options(options: &FrameInferenceOptions) -> Result<()> {
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

fn open_detection_batch_sink(
    options: &BatchOutputOptions,
) -> Result<(DetectionBatchSink, Option<f64>)> {
    match &options.destination {
        BatchOutputDestination::JsonlFile { path } => {
            if let Some(parent) = path.parent() {
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
                .open(path)
                .with_context(|| {
                    format!(
                        "failed to create detection batch output {} without overwriting",
                        path.display()
                    )
                })?;
            Ok((DetectionBatchSink::Jsonl(BufWriter::new(file)), None))
        }
        BatchOutputDestination::LoopbackTcp {
            address,
            maximum_message_bytes,
            connect_timeout,
            write_timeout,
        } => {
            ensure!(
                address.ip().is_loopback(),
                "batch TCP address must be loopback"
            );
            let connect_start = Instant::now();
            let stream = TcpStream::connect_timeout(address, *connect_timeout)
                .with_context(|| format!("failed to connect to batch receiver at {address}"))?;
            stream
                .set_write_timeout(Some(*write_timeout))
                .context("failed to configure batch receiver write timeout")?;
            stream
                .set_nodelay(true)
                .context("failed to enable low-latency batch TCP writes")?;
            Ok((
                DetectionBatchSink::LoopbackTcp {
                    stream,
                    maximum_message_bytes: *maximum_message_bytes,
                },
                Some(elapsed_ms(connect_start)),
            ))
        }
    }
}

fn open_scene_snapshot_output(
    options: &SceneSnapshotOutputOptions,
    batch_options: &BatchOutputOptions,
) -> Result<SceneSnapshotOutputRuntime> {
    match &options.destination {
        SceneSnapshotOutputDestination::JsonlFile { .. } => ensure!(
            options.delivery_schema_version.is_none(),
            "native public-state delivery schema selection is supported only for TCP output"
        ),
        SceneSnapshotOutputDestination::LoopbackTcp { .. } => ensure!(
            options.delivery_schema_version == Some(PUBLIC_STATE_DELIVERY_SCHEMA_VERSION),
            "scene TCP output requires the supported public-state delivery schema version 1"
        ),
    }
    let (
        destination,
        format,
        sink,
        connect_ms,
        maximum_message_bytes,
        connect_timeout_ms,
        write_timeout_ms,
    ) = match &options.destination {
        SceneSnapshotOutputDestination::JsonlFile { path } => {
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
            (
                path.display().to_string(),
                "newline_delimited_scene_snapshot_v1",
                SceneSnapshotSink::Jsonl(BufWriter::new(file)),
                None,
                None,
                None,
                None,
            )
        }
        SceneSnapshotOutputDestination::LoopbackTcp {
            address,
            maximum_message_bytes,
            connect_timeout,
            write_timeout,
            reconnect_initial_delay,
            reconnect_maximum_delay,
        } => {
            ensure!(
                address.ip().is_loopback(),
                "scene TCP address must be loopback"
            );
            let writer = LatestSceneWriter::start(SceneTransportConfig {
                address: *address,
                maximum_message_bytes: *maximum_message_bytes,
                public_state_delivery_enabled: true,
                connect_timeout: *connect_timeout,
                write_timeout: *write_timeout,
                reconnect_initial_delay: *reconnect_initial_delay,
                reconnect_maximum_delay: *reconnect_maximum_delay,
            })?;
            (
                format!("tcp://{address}"),
                "length_framed_json_public_state_delivery_v1",
                SceneSnapshotSink::LoopbackTcp {
                    writer,
                    maximum_message_bytes: *maximum_message_bytes,
                },
                None,
                Some(*maximum_message_bytes),
                Some(connect_timeout.as_secs_f64() * 1000.0),
                Some(write_timeout.as_secs_f64() * 1000.0),
            )
        }
    };
    let lifecycle = match &options.destination {
        SceneSnapshotOutputDestination::JsonlFile { .. } => None,
        SceneSnapshotOutputDestination::LoopbackTcp { .. } => {
            ensure!(
                options.delivery_schema_version == Some(PUBLIC_STATE_DELIVERY_SCHEMA_VERSION),
                "scene TCP output requires the supported public-state delivery schema version 1"
            );
            let mut lifecycle = SceneStateLifecycle::new(
                StateLifecycleConfig::default(),
                PublicStateConfig::default(),
            )
            .context("failed to create replay public-state lifecycle")?;
            lifecycle
                .activate_source(
                    batch_options.engine_instance_id.clone(),
                    batch_options.source_session_id.clone(),
                )
                .context("failed to activate replay public-state producer identity")?;
            Some(lifecycle)
        }
    };
    Ok(SceneSnapshotOutputRuntime {
        destination,
        format,
        connect_ms,
        maximum_message_bytes,
        connect_timeout_ms,
        write_timeout_ms,
        tracker: SceneTrackerV1::new(options.tracking.clone()),
        lifecycle,
        sink,
        snapshot_count: 0,
        total_tracking_ms: 0.0,
        total_encode_enqueue_ms: 0.0,
        transport_stats: None,
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

fn write_scene_snapshot(
    sink: &mut SceneSnapshotSink,
    snapshot: &SceneSnapshotV1,
    current_state: Option<&CurrentSceneState>,
) -> Result<()> {
    let json = encode_scene_snapshot_v1(snapshot).context("scene snapshot failed validation")?;
    match sink {
        SceneSnapshotSink::Jsonl(writer) => write_scene_snapshot_jsonl(writer, &json),
        SceneSnapshotSink::LoopbackTcp {
            writer,
            maximum_message_bytes,
        } => {
            let delivery = PublicStateDeliveryV1::new(snapshot, current_state);
            let delivery_json = serde_json::to_vec(&delivery)
                .context("failed to encode public-state delivery v1")?;
            ensure!(
                delivery_json.len() <= *maximum_message_bytes,
                "public-state delivery exceeds configured maximum message size"
            );
            writer.submit(
                snapshot.transport_sequence,
                &snapshot.engine_instance_id,
                &snapshot.source.session_id,
                delivery_json,
            )
        }
    }
}

fn summarize_scene_snapshot_output(
    options: &SceneSnapshotOutputOptions,
    runtime: &SceneSnapshotOutputRuntime,
) -> SceneSnapshotOutputReport {
    let config = &options.tracking;
    let sample_count = runtime.snapshot_count.max(1) as f64;
    SceneSnapshotOutputReport {
        destination: runtime.destination.clone(),
        format: runtime.format,
        mode: if runtime.maximum_message_bytes.is_some() {
            "bounded_latest_only_asynchronous_native_public_state_delivery_during_paced_replay"
        } else {
            "synchronous_file_export_during_paced_replay"
        },
        snapshot_count: runtime.snapshot_count,
        connect_ms: runtime.connect_ms,
        maximum_message_bytes: runtime.maximum_message_bytes,
        connect_timeout_ms: runtime.connect_timeout_ms,
        write_timeout_ms: runtime.write_timeout_ms,
        transport: runtime.transport_stats.clone(),
        ball_class_id: config.ball_class_id(),
        cue_class_id: config.cue_class_id(),
        camera_profile_id: config.camera_profile_id().to_owned(),
        association_gate_px: config.association_gate_px(),
        association_prediction_horizon_ms: config.association_prediction_horizon().as_secs_f64()
            * 1000.0,
        lost_after_ms: config.lost_after().as_secs_f64() * 1000.0,
        maximum_track_age_ms: config.maximum_track_age().as_secs_f64() * 1000.0,
        maximum_active_tracks: config.maximum_active_tracks(),
        recent_observation_capacity: config.recent_observation_capacity(),
        configuration_revision: config.configuration_revision(),
        rectification_revision: config.rectification_revision(),
        mean_tracking_ms: runtime.total_tracking_ms / sample_count,
        mean_encode_enqueue_ms: runtime.total_encode_enqueue_ms / sample_count,
        source_time: "synthetic_monotonic_frame_arrival",
    }
}

fn batch_destination_label(destination: &BatchOutputDestination) -> String {
    match destination {
        BatchOutputDestination::JsonlFile { path } => path.display().to_string(),
        BatchOutputDestination::LoopbackTcp { address, .. } => format!("tcp://{address}"),
    }
}

fn batch_output_format(destination: &BatchOutputDestination) -> &'static str {
    match destination {
        BatchOutputDestination::JsonlFile { .. } => "newline_delimited_json",
        BatchOutputDestination::LoopbackTcp { .. } => "length_framed_json",
    }
}

fn batch_output_mode(destination: &BatchOutputDestination, is_paced: bool) -> &'static str {
    match (destination, is_paced) {
        (BatchOutputDestination::JsonlFile { .. }, false) => {
            "synchronous_export_during_unpaced_static_replay"
        }
        (BatchOutputDestination::LoopbackTcp { .. }, false) => {
            "synchronous_loopback_delivery_during_unpaced_static_replay"
        }
        (BatchOutputDestination::JsonlFile { .. }, true) => {
            "synchronous_export_during_paced_latest_frame_replay"
        }
        (BatchOutputDestination::LoopbackTcp { .. }, true) => {
            "synchronous_loopback_delivery_during_paced_latest_frame_replay"
        }
    }
}

fn batch_maximum_message_bytes(destination: &BatchOutputDestination) -> Option<usize> {
    match destination {
        BatchOutputDestination::JsonlFile { .. } => None,
        BatchOutputDestination::LoopbackTcp {
            maximum_message_bytes,
            ..
        } => Some(*maximum_message_bytes),
    }
}

fn batch_connect_timeout_ms(destination: &BatchOutputDestination) -> Option<f64> {
    match destination {
        BatchOutputDestination::JsonlFile { .. } => None,
        BatchOutputDestination::LoopbackTcp {
            connect_timeout, ..
        } => Some(connect_timeout.as_secs_f64() * 1000.0),
    }
}

fn batch_write_timeout_ms(destination: &BatchOutputDestination) -> Option<f64> {
    match destination {
        BatchOutputDestination::JsonlFile { .. } => None,
        BatchOutputDestination::LoopbackTcp { write_timeout, .. } => {
            Some(write_timeout.as_secs_f64() * 1000.0)
        }
    }
}

fn write_detection_batch(sink: &mut DetectionBatchSink, batch: &DetectionBatchV1) -> Result<()> {
    match sink {
        DetectionBatchSink::Jsonl(writer) => write_detection_batch_line(writer, batch),
        DetectionBatchSink::LoopbackTcp {
            stream,
            maximum_message_bytes,
        } => {
            let json =
                encode_detection_batch_v1(batch).context("detection batch failed validation")?;
            write_framed_message(stream, &json, *maximum_message_bytes)
                .context("failed to send length-framed detection batch")?;
            Ok(())
        }
    }
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

/// Apply the same current-state lifecycle as live capture, anchoring its
/// receipt clock to replay-local arrival rather than inference completion.
/// `now` is sampled after tracking and before delivery, so freshness includes
/// image preparation, inference, and scene tracking latency.
fn ingest_replay_scene(
    lifecycle: &mut SceneStateLifecycle,
    snapshot: &SceneSnapshotV1,
    clock_origin: Instant,
    source_arrived_at: Instant,
    now: Instant,
) -> Result<Option<CurrentSceneState>> {
    let received_monotonic_ns = replay_monotonic_ns(clock_origin, source_arrived_at)?;
    let now_monotonic_ns = replay_monotonic_ns(clock_origin, now)?;
    lifecycle
        .ingest(
            snapshot,
            received_monotonic_ns,
            now_monotonic_ns,
            None,
            None,
        )
        .context("failed to apply replay scene to Rust public-state lifecycle")?;
    lifecycle
        .current(now_monotonic_ns)
        .context("failed to read replay public-state lifecycle")
}

fn replay_monotonic_ns(origin: Instant, at: Instant) -> Result<u128> {
    at.checked_duration_since(origin)
        .context("replay-local monotonic timestamp precedes its clock origin")
        .map(|elapsed| elapsed.as_nanos())
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

fn collect_images(path: &Path) -> Result<Vec<PathBuf>> {
    ensure!(
        path.exists(),
        "input path does not exist: {}",
        path.display()
    );
    let mut images = Vec::new();
    if path.is_file() {
        ensure!(
            is_supported_image(path),
            "input file must be a .png, .jpg, or .jpeg image: {}",
            path.display()
        );
        images.push(path.to_path_buf());
    } else {
        collect_directory_images(path, &mut images)?;
        images.sort();
    }
    ensure!(
        !images.is_empty(),
        "no PNG/JPEG images found under {}",
        path.display()
    );
    Ok(images)
}

fn collect_directory_images(directory: &Path, images: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_directory_images(&path, images)?;
        } else if is_supported_image(&path) {
            images.push(path);
        }
    }
    Ok(())
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            SUPPORTED_IMAGE_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
                || extension.eq_ignore_ascii_case("jpeg")
        })
        .unwrap_or(false)
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
        return OpenVinoSession::new(onnx_path, intra_op_threads).map(InferenceSession::Openvino);
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
    fn new(onnx_path: &Path, intra_op_threads: Option<usize>) -> Result<Self> {
        let onnx_path = onnx_path
            .to_str()
            .context("OpenVINO requires a UTF-8 ONNX model path")?;
        load_openvino_c_library()?;
        let mut core = Core::new().context(
            "failed to load OpenVINO Runtime; install the matching OpenVINO runtime and set its platform library path",
        )?;
        if let Some(threads) = intra_op_threads {
            let thread_count = threads.to_string();
            core.set_property(
                &DeviceType::CPU,
                &RwPropertyKey::InferenceNumThreads,
                &thread_count,
            )
            .context("failed to configure OpenVINO CPU inference thread count")?;
        }
        let model = core
            .read_model_from_file(onnx_path, "")
            .with_context(|| format!("OpenVINO could not read ONNX model {onnx_path}"))?;
        let mut compiled_model = core
            .compile_model(&model, DeviceType::CPU)
            .context("OpenVINO could not compile the ONNX model for CPU")?;
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
            _compiled_model: compiled_model,
            _core: core,
        })
    }
}

fn validate_session(
    session: &InferenceSession,
    image_size: u32,
    class_names: &[String],
) -> Result<(Vec<i64>, Vec<i64>)> {
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
    Ok((input_shape, output_shape))
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
        ensure!(
            options.retained_provider_profile.is_none(),
            "retained ONNX Runtime provider profiles are unavailable for the native OpenVINO backend"
        );
        let init_start = Instant::now();
        let mut session = create_session(environment, options, None)?;
        let (_, output_shape) = validate_session(&session, options.image_size, class_names)?;
        ensure!(
            output_shape == expected_output_shape,
            "OpenVINO output signature is unexpected: {output_shape:?}"
        );
        let session_init_ms = elapsed_ms(init_start);
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
                observed_providers: vec!["OpenVINO CPU (native)".to_owned()],
                event_count: 0,
                node_counts: BTreeMap::new(),
                retained_profile_path: None,
            },
            session_init_ms,
            inference_ms,
            sample,
        ));
    }

    let temporary_directory =
        tempfile::tempdir().context("failed to create ORT profile scratch directory")?;
    let scratch_profile = temporary_directory.path().join("provider-profile.json");
    let profile_path = options
        .retained_provider_profile
        .as_deref()
        .unwrap_or(&scratch_profile);
    if let Some(parent) = options
        .retained_provider_profile
        .as_ref()
        .and_then(|path| path.parent())
    {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create provider trace directory {}",
                parent.display()
            )
        })?;
    }

    let init_start = Instant::now();
    let mut session = create_session(environment, options, Some(profile_path))?;
    let (_, output_shape) = validate_session(&session, options.image_size, class_names)?;
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

    let retained_profile_path = if options.retained_provider_profile.is_some() {
        Some(profile_path)
    } else {
        None
    };
    Ok((
        ProviderEvidence {
            observed_providers,
            event_count,
            node_counts,
            retained_profile_path,
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
                .context("OpenVINO CPU inference failed")?;
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

fn load_and_prepare(path: &Path, image_size: u32) -> Result<TimedInput> {
    let (image, decode_ms) = load_rgb_image(path)?;
    prepare_decoded_image(image, decode_ms, image_size)
}

fn load_rgb_image(path: &Path) -> Result<(RgbImage, f64)> {
    let decode_start = Instant::now();
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open input image {}", path.display()))?
        .with_guessed_format()
        .context("failed to identify input image format")?
        .decode()
        .with_context(|| format!("failed to decode input image {}", path.display()))?
        .to_rgb8();
    Ok((image, elapsed_ms(decode_start)))
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

fn summarize_calibration(
    options: &CalibrationAuditOptions,
    applied_image_space: ImageSpace,
    results: &[ImageResult],
) -> CalibrationAuditReport {
    let calibrated_frame_count = results
        .iter()
        .filter(|result| {
            result
                .calibration
                .as_ref()
                .is_some_and(|frame| matches!(frame.status, CalibrationStatus::Valid))
        })
        .count();
    let calibrated_detection_count = results
        .iter()
        .filter_map(|result| result.calibration.as_ref())
        .filter_map(|frame| frame.measurements.as_ref())
        .map(Vec::len)
        .sum();

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
        frame_count: results.len(),
        calibrated_frame_count,
        unavailable_frame_count: results.len() - calibrated_frame_count,
        calibrated_detection_count,
    }
}

#[allow(clippy::too_many_arguments)]
fn summarize_timings(
    results: &[ImageResult],
    provider_probe_session_init_ms: f64,
    provider_probe_inference_ms: f64,
    measurement_session_init_ms: f64,
    warmup_total_ms: f64,
    total_recorded_image_ms: f64,
    batch_export_ms: Option<f64>,
    paced_replay: Option<PacedReplayReport>,
) -> RunTimings {
    let count = results.len().max(1) as f64;
    let mean = |select: fn(&ImageTimings) -> f64| {
        results
            .iter()
            .map(|result| select(&result.timings))
            .sum::<f64>()
            / count
    };
    let optional_mean = |select: fn(&ImageTimings) -> Option<f64>| {
        let samples = results
            .iter()
            .filter_map(|result| select(&result.timings))
            .collect::<Vec<_>>();
        (!samples.is_empty()).then(|| samples.iter().sum::<f64>() / samples.len() as f64)
    };
    RunTimings {
        image_count: results.len(),
        provider_probe_session_init_ms,
        provider_probe_inference_ms,
        measurement_session_init_ms,
        warmup_total_ms,
        total_recorded_image_ms,
        summed_image_pipeline_ms: results
            .iter()
            .map(|result| result.timings.end_to_end_ms)
            .sum(),
        mean_image_decode_ms: mean(|timing| timing.image_decode_ms),
        mean_preprocess_ms: mean(|timing| timing.preprocess_ms),
        mean_inference_ms: mean(|timing| timing.inference_ms),
        mean_decode_and_nms_ms: mean(|timing| timing.decode_and_nms_ms),
        mean_end_to_end_ms: mean(|timing| timing.end_to_end_ms),
        mean_scene_tracking_ms: optional_mean(|timing| timing.scene_tracking_ms),
        mean_scene_encode_enqueue_ms: optional_mean(|timing| timing.scene_encode_enqueue_ms),
        mean_calibration_ms: results
            .iter()
            .filter_map(|result| result.timings.calibration_ms)
            .reduce(|total, sample| total + sample)
            .map(|total| {
                let sample_count = results
                    .iter()
                    .filter(|result| result.timings.calibration_ms.is_some())
                    .count();
                total / sample_count as f64
            }),
        source_to_calibration_complete_ms: results
            .iter()
            .any(|result| result.calibration.is_some())
            .then(|| {
                percentile_summary(
                    results
                        .iter()
                        .filter_map(|result| result.source_to_calibration_complete_ms)
                        .collect(),
                )
            }),
        batch_export_ms,
        paced_replay,
    }
}

fn write_nchw_float32(path: &Path, values: &[f32]) -> Result<()> {
    use std::io::{BufWriter, Write};

    let mut file = fs::File::create(path)
        .with_context(|| format!("failed to create preprocessed tensor {}", path.display()))?;
    let mut writer = BufWriter::new(&mut file);
    let mut bytes = Vec::with_capacity(1024 * std::mem::size_of::<f32>());
    for chunk in values.chunks(1024) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer
            .write_all(&bytes)
            .with_context(|| format!("failed to write preprocessed tensor {}", path.display()))?;
    }
    Ok(())
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::{
        build_detection_batch, calibration_report_for_batch, checked_duration_from_seconds,
        completed_after_source_deadline, expected_anchor_count, find_versioned_openvino_c_library,
        frames_per_second, ingest_replay_scene, latest_due_source_index, load_and_verify_manifest,
        open_detection_batch_sink, open_scene_snapshot_output, parse_provider_profile,
        percentile_summary, produce_paced_frames, replay_frame_period, run_bounded_command,
        sha256_file, timeline_media_duration, validate_batch_identifier,
        validate_frame_inference_options, validate_inference_parameters,
        validate_replay_frame_budget, validate_scene_calibration_context,
        versioned_openvino_library_version, write_detection_batch_line, write_nchw_float32,
        write_scene_snapshot, write_scene_snapshot_line, AuditOptions, BatchOutputDestination,
        BatchOutputOptions, CalibrationAuditOptions, CalibrationStatus, DecodedReplayFrame,
        FrameInferenceEngine, FrameInferenceOptions, PacedReplayMailbox,
        PacedReplayMeasurementWindow, PacedReplayRateCounts, ProviderSelection,
        SceneSnapshotOutputDestination, SceneSnapshotOutputOptions, VideoTimeline,
    };
    use crate::detection_batch::{decode_detection_batch_v1, DetectionBatchV1, ImageSpace};
    use crate::inference::Detection;
    use crate::public_state::PublicStateConfig;
    use crate::public_state_lifecycle::{
        SceneStateLifecycle, StateFreshness, StateLifecycleConfig,
    };
    use crate::tracking::TrackingConfigV1;
    use image::{Rgb, RgbImage};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::fs;
    use std::net::{SocketAddr, TcpListener};
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
                retained_provider_profile: None,
                dump_preprocessed_directory: None,
                source_image_space: ImageSpace::Raw,
                detection_batch_output: None,
                calibration: None,
                scene_snapshot_output: None,
                replay_rate_fps: None,
                replay_repeat_count: 1,
                maximum_replay_frames: None,
            },
            BatchOutputOptions {
                destination: super::BatchOutputDestination::JsonlFile {
                    path: PathBuf::from("batches.jsonl"),
                },
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
    fn calibration_report_distinguishes_valid_mismatch_and_successful_empty_frames() {
        let populated_batch = calibrated_batch(SUCCESS_BATCH_FIXTURE);
        let valid =
            calibration_report_for_batch(&populated_batch, &calibration_options(3)).unwrap();
        assert!(matches!(valid.status, CalibrationStatus::Valid));
        assert_eq!(valid.measurements.as_ref().unwrap().len(), 1);
        assert_eq!(
            valid.measurements.as_ref().unwrap()[0].center_table_mm.x_mm,
            0.0
        );
        assert_eq!(
            valid.measurements.as_ref().unwrap()[0].center_table_mm.y_mm,
            0.0
        );

        let unavailable =
            calibration_report_for_batch(&populated_batch, &calibration_options(4)).unwrap();
        assert!(matches!(unavailable.status, CalibrationStatus::Unavailable));
        assert!(unavailable
            .reason
            .as_deref()
            .unwrap()
            .contains("CalibrationRevisionMismatch"));
        assert!(unavailable.measurements.is_none());

        let empty_batch = calibrated_batch(EMPTY_BATCH_FIXTURE);
        let empty = calibration_report_for_batch(&empty_batch, &calibration_options(3)).unwrap();
        assert!(matches!(empty.status, CalibrationStatus::Valid));
        assert_eq!(empty.measurements, Some(Vec::new()));
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
    fn paced_replay_period_rejects_invalid_or_unrepresentable_rates() {
        assert_eq!(
            replay_frame_period(25.0).unwrap(),
            Duration::from_millis(40)
        );
        assert!(replay_frame_period(0.0).is_err());
        assert!(replay_frame_period(f64::INFINITY).is_err());
        assert!(replay_frame_period(1.0e12).is_err());
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
        assert!(short.unwrap_err().to_string().contains("expected 9 bytes"));

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
    fn paced_rates_use_matching_event_counts_and_exclude_slow_drain() {
        let start = Instant::now();
        let window = PacedReplayMeasurementWindow::new(start, Duration::from_millis(100)).unwrap();
        let mut counts = PacedReplayRateCounts::default();

        // Three images finish decoding during the source window. Two more are
        // decoded after its fixed end while the paced producer catches up.
        for offset_ms in [0, 20, 99, 100, 140] {
            counts.record_decoded(start + Duration::from_millis(offset_ms), window);
        }
        // Two inference results complete inside the same interval. The later
        // results belong to drain, even though they contribute to total
        // processed_frame_count and the latency distribution.
        for offset_ms in [40, 80, 100, 150] {
            counts.record_inference_complete(start + Duration::from_millis(offset_ms), window);
        }

        assert_eq!(counts.decoded_in_measurement_window, 3);
        assert_eq!(counts.decoded_after_measurement_window, 2);
        assert_eq!(counts.inference_complete_in_measurement_window, 2);
        assert_eq!(counts.inference_complete_after_measurement_window, 2);

        let received_fps = frames_per_second(counts.decoded_in_measurement_window, window);
        let processed_fps =
            frames_per_second(counts.inference_complete_in_measurement_window, window);
        assert_eq!(received_fps, 30.0);
        assert_eq!(processed_fps, 20.0);
        assert!(processed_fps <= received_fps);
    }

    #[test]
    fn paced_replay_frame_budget_requires_a_cap_for_repeated_input() {
        assert_eq!(validate_replay_frame_budget(2, 3, Some(6)).unwrap(), 6);
        assert_eq!(validate_replay_frame_budget(2, 1, None).unwrap(), 2);
        assert!(validate_replay_frame_budget(2, 3, None).is_err());
        assert!(validate_replay_frame_budget(2, 3, Some(5)).is_err());
        assert!(validate_replay_frame_budget(2, 0, Some(10)).is_err());
        assert!(validate_replay_frame_budget(0, 1, Some(10)).is_err());
        assert!(validate_replay_frame_budget(2, 3, Some(0)).is_err());
        assert!(validate_replay_frame_budget(usize::MAX, 2, Some(10)).is_err());
    }

    #[test]
    fn paced_replay_repeats_recorded_images_with_fresh_source_indexes() {
        let directory = tempfile::tempdir().unwrap();
        let image_paths = (0..2)
            .map(|index| {
                let path = directory.path().join(format!("frame-{index}.png"));
                RgbImage::from_pixel(2, 2, Rgb([index as u8, 0, 0]))
                    .save(&path)
                    .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let image_hashes = vec!["hash-a".to_owned(), "hash-b".to_owned()];
        let mailbox = PacedReplayMailbox::default();

        produce_paced_frames(
            &mailbox,
            &image_paths,
            &image_hashes,
            3,
            6,
            Instant::now(),
            Duration::ZERO,
        )
        .unwrap();
        mailbox.finish(None);

        let latest = mailbox.take_latest().unwrap().unwrap();
        assert_eq!(latest.source_index, 5);
        assert_eq!(latest.path, image_paths[1]);
        assert_eq!(latest.encoded_image_sha256, "hash-b");
        assert_eq!(latest.image.get_pixel(0, 0), &Rgb([1, 0, 0]));
        assert!(mailbox.take_latest().unwrap().is_none());

        let stats = mailbox.snapshot().unwrap();
        assert_eq!(stats.decoded_frame_count, 6);
        assert_eq!(stats.replaced_pending_frame_count, 5);
    }

    #[test]
    fn paced_replay_schedule_stays_continuous_across_repeated_corpus_cycles() {
        let directory = tempfile::tempdir().unwrap();
        let image_paths = (0..2)
            .map(|index| {
                let path = directory.path().join(format!("frame-{index}.png"));
                RgbImage::from_pixel(2, 2, Rgb([index as u8, 0, 0]))
                    .save(&path)
                    .unwrap();
                path
            })
            .collect::<Vec<_>>();
        let image_hashes = vec!["hash-a".to_owned(), "hash-b".to_owned()];
        let mailbox = PacedReplayMailbox::default();
        let replay_started = Instant::now();
        let frame_period = Duration::from_millis(100);
        let producer_mailbox = mailbox.clone();
        let producer_paths = image_paths.clone();
        let producer_hashes = image_hashes.clone();
        let producer = std::thread::spawn(move || {
            let result = produce_paced_frames(
                &producer_mailbox,
                &producer_paths,
                &producer_hashes,
                3,
                6,
                replay_started,
                frame_period,
            );
            producer_mailbox.finish(result.as_ref().err().map(ToString::to_string));
            result
        });

        let mut frames = Vec::new();
        while let Some(frame) = mailbox.take_latest().unwrap() {
            frames.push(frame);
        }
        producer.join().unwrap().unwrap();

        assert_eq!(frames.len(), 6);
        for (source_index, frame) in frames.iter().enumerate() {
            let corpus_index = source_index % image_paths.len();
            assert_eq!(frame.source_index, source_index);
            assert_eq!(frame.path, image_paths[corpus_index]);
            assert_eq!(frame.encoded_image_sha256, image_hashes[corpus_index]);
            assert_eq!(
                frame.scheduled_at.duration_since(replay_started),
                frame_period * source_index as u32
            );
        }
        assert_eq!(
            frames[4]
                .scheduled_at
                .duration_since(frames[3].scheduled_at),
            frame_period
        );
        let stats = mailbox.snapshot().unwrap();
        assert_eq!(stats.decoded_frame_count, 6);
        assert_eq!(stats.replaced_pending_frame_count, 0);
    }

    #[test]
    fn latest_frame_mailbox_replaces_stale_pending_input_and_counts_misses() {
        let mailbox = PacedReplayMailbox::default();
        let base = Instant::now();
        let period = Duration::from_millis(33);
        for (source_index, lateness) in [Duration::from_millis(35), Duration::from_millis(2)]
            .into_iter()
            .enumerate()
        {
            mailbox
                .offer(
                    DecodedReplayFrame {
                        source_index,
                        path: PathBuf::from(format!("frame-{source_index}.png")),
                        encoded_image_sha256: format!("hash-{source_index}"),
                        image: image::RgbImage::new(1, 1),
                        decode_ms: 0.0,
                        scheduled_at: base,
                        source_arrived_at: base + lateness,
                    },
                    period,
                )
                .unwrap();
        }
        let latest = mailbox.take_latest().unwrap().unwrap();
        assert_eq!(latest.source_index, 1);
        assert_eq!(latest.path, PathBuf::from("frame-1.png"));
        mailbox.finish(None);
        assert!(mailbox.take_latest().unwrap().is_none());

        let stats = mailbox.snapshot().unwrap();
        assert_eq!(stats.decoded_frame_count, 2);
        assert_eq!(stats.replaced_pending_frame_count, 1);
        assert_eq!(stats.source_schedule_deadline_misses, 1);
        assert_eq!(stats.maximum_source_schedule_lateness_ms, 35.0);
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
    fn optional_tensor_export_is_little_endian_float32() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("input.f32le");
        write_nchw_float32(&path, &[1.0, -2.5]).unwrap();
        assert_eq!(
            std::fs::read(path).unwrap(),
            [1.0_f32.to_le_bytes(), (-2.5_f32).to_le_bytes()].concat()
        );
    }

    #[test]
    fn manifest_requires_matching_source_and_export_hashes() {
        let directory = tempfile::tempdir().unwrap();
        let onnx_path = directory.path().join("model.onnx");
        let source_path = directory.path().join("source.pt");
        fs::write(&onnx_path, b"onnx test bytes").unwrap();
        fs::write(&source_path, b"source checkpoint test bytes").unwrap();
        let manifest = serde_json::json!({
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
            "OpenVINO CPU (native)"
        );
        assert_eq!(
            ProviderSelection::Openvino.binding(),
            (
                "openvino-rs",
                "0.11.1",
                "Native OpenVINO CPU backend selected and invoked; this API does not expose per-node fallback assignment, so node counts are unavailable"
            )
        );
        assert_eq!(
            ProviderSelection::Cpu.binding().0,
            "ort",
            "portable CPU inference remains on ONNX Runtime"
        );
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
    fn paced_tcp_scene_output_wraps_the_raw_snapshot_and_native_current_state() {
        const MAXIMUM_MESSAGE_BYTES: usize = 64 * 1024;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let receiver = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let hello: Value = serde_json::from_slice(
                &crate::framing::read_framed_message(&mut stream, MAXIMUM_MESSAGE_BYTES).unwrap(),
            )
            .unwrap();
            assert_eq!(hello["protocol_version"], 2);
            let ready = serde_json::to_vec(&json!({
                "type": "ready",
                "protocol_version": 2,
                "last_transport_sequence": null,
            }))
            .unwrap();
            crate::framing::write_framed_message(&mut stream, &ready, MAXIMUM_MESSAGE_BYTES)
                .unwrap();
            serde_json::from_slice::<Value>(
                &crate::framing::read_framed_message(&mut stream, MAXIMUM_MESSAGE_BYTES).unwrap(),
            )
            .unwrap()
        });

        let snapshot = crate::decode_scene_snapshot_v1(TRACKED_SCENE_FIXTURE.as_bytes()).unwrap();
        let (_, mut batch_output) = batch_options();
        batch_output.engine_instance_id = snapshot.engine_instance_id.clone();
        batch_output.source_session_id = snapshot.source.session_id.clone();
        let tracking = TrackingConfigV1::new(
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
        .unwrap();
        let options = SceneSnapshotOutputOptions {
            destination: SceneSnapshotOutputDestination::LoopbackTcp {
                address,
                maximum_message_bytes: MAXIMUM_MESSAGE_BYTES,
                connect_timeout: Duration::from_millis(500),
                write_timeout: Duration::from_millis(500),
                reconnect_initial_delay: Duration::from_millis(10),
                reconnect_maximum_delay: Duration::from_millis(50),
            },
            tracking,
            delivery_schema_version: Some(1),
        };
        let mut output = open_scene_snapshot_output(&options, &batch_output).unwrap();
        let raw_json = crate::encode_scene_snapshot_v1(&snapshot).unwrap();
        let raw_value: Value = serde_json::from_slice(&raw_json).unwrap();
        let clock_origin = Instant::now();
        let state = ingest_replay_scene(
            output.lifecycle.as_mut().unwrap(),
            &snapshot,
            clock_origin,
            clock_origin + Duration::from_millis(1),
            clock_origin + Duration::from_millis(2),
        )
        .unwrap()
        .unwrap();
        let expected_state = serde_json::to_value(&state).unwrap();

        write_scene_snapshot(&mut output.sink, &snapshot, Some(&state)).unwrap();
        output.finish();
        let delivery = receiver.join().unwrap();

        assert_eq!(delivery["delivery_type"], "billiards_public_state_delivery");
        assert_eq!(delivery["delivery_schema_version"], 1);
        assert_eq!(delivery["source_snapshot"], raw_value);
        assert_eq!(delivery["current_state"], expected_state);
        assert_eq!(delivery["current_state"]["freshness"], "fresh");
    }

    #[test]
    fn replay_arrival_before_slow_inference_expires_old_coordinates() {
        let snapshot = crate::decode_scene_snapshot_v1(TRACKED_SCENE_FIXTURE.as_bytes()).unwrap();
        let mut lifecycle = SceneStateLifecycle::new(
            StateLifecycleConfig {
                stale_after_ns: 20,
                expire_after_ns: 40,
                maximum_balls: 64,
            },
            PublicStateConfig::default(),
        )
        .unwrap();
        lifecycle
            .activate_source(
                snapshot.engine_instance_id.clone(),
                snapshot.source.session_id.clone(),
            )
            .unwrap();
        let clock_origin = Instant::now();
        let state = ingest_replay_scene(
            &mut lifecycle,
            &snapshot,
            clock_origin,
            clock_origin + Duration::from_nanos(10),
            clock_origin + Duration::from_nanos(70),
        )
        .unwrap()
        .unwrap();

        assert_eq!(state.freshness, StateFreshness::Expired);
        assert_eq!(state.localization_age_ns, Some(60));
        assert!(state.payload.balls.is_empty());
        assert_eq!(
            state.payload.observation_status["freshness_status"],
            "expired"
        );
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
            delivery_schema_version: None,
        };
        let (_, batch_output) = batch_options();
        let output = open_scene_snapshot_output(&options, &batch_output).unwrap();
        drop(output);
        assert!(open_scene_snapshot_output(&options, &batch_output).is_err());
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
            destination: super::BatchOutputDestination::JsonlFile { path: path.clone() },
            engine_instance_id: "engine-test".to_owned(),
            source_session_id: "session-test".to_owned(),
        };

        assert!(open_detection_batch_sink(&options).is_err());
        assert_eq!(fs::read(path).unwrap(), b"preserve this data");
    }

    #[test]
    fn loopback_sink_sends_a_complete_length_framed_batch() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let receiver = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let bytes = crate::framing::read_framed_message(&mut stream, 64 * 1024).unwrap();
            decode_detection_batch_v1(&bytes).unwrap()
        });

        let (audit, _) = batch_options();
        let destination = BatchOutputDestination::LoopbackTcp {
            address,
            maximum_message_bytes: 64 * 1024,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
        };
        let output = BatchOutputOptions {
            destination,
            engine_instance_id: "engine-test-01".to_owned(),
            source_session_id: "recording-test-01".to_owned(),
        };
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
            &audit,
            &output,
            origin,
            origin,
            origin,
            None,
        )
        .unwrap();
        let (mut sink, connect_ms) = open_detection_batch_sink(&output).unwrap();
        assert!(connect_ms.is_some());
        super::write_detection_batch(&mut sink, &batch).unwrap();
        drop(sink);

        let received = receiver.join().unwrap();
        assert_eq!(received.batch_sequence, 1);
        assert_eq!(received.source.session_id, "recording-test-01");
    }

    #[test]
    fn loopback_sink_rejects_non_loopback_destinations_before_connecting() {
        let options = BatchOutputOptions {
            destination: BatchOutputDestination::LoopbackTcp {
                address: "192.0.2.1:1234".parse::<SocketAddr>().unwrap(),
                maximum_message_bytes: 64 * 1024,
                connect_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_secs(1),
            },
            engine_instance_id: "engine-test-01".to_owned(),
            source_session_id: "recording-test-01".to_owned(),
        };
        assert!(open_detection_batch_sink(&options).is_err());
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
                onnx_path: model_path.clone(),
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
    fn live_frame_options_share_the_audit_inference_parameter_validation() {
        assert!(validate_inference_parameters(32, 0.4, 0.45, 1, Some(1)).is_ok());
        assert!(validate_inference_parameters(31, 0.4, 0.45, 1, Some(1)).is_err());
        assert!(validate_inference_parameters(32, f32::NAN, 0.45, 1, Some(1)).is_err());
        assert!(validate_inference_parameters(32, 0.4, 0.45, 0, Some(1)).is_err());
        assert!(validate_inference_parameters(32, 0.4, 0.45, 1, Some(0)).is_err());

        let scratch = tempfile::tempdir().unwrap();
        let options = FrameInferenceOptions {
            onnx_path: scratch.path().join("missing.onnx"),
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
