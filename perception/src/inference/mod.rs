//! Verified two-class YOLO inference and recorded-image replay.
//!
//! The reusable frame engine deliberately fails closed on model signatures
//! that differ from the verified export contract. The sibling
//! `capture_pipeline` module can compose it with the device-independent frame
//! receiver and bounded scene writer; no platform camera adapter or live
//! provider startup is selected or qualified here.

mod decode;
mod preprocess;
mod runtime;

pub use decode::{decode_yolo_v8_output, Detection};
pub use preprocess::{prepare_rgb8, prepare_rgb_image, LetterboxGeometry, PreparedInput};
pub use runtime::{
    run_audit, run_video_audit, AuditOptions, AuditReport, BatchOutputDestination,
    BatchOutputOptions, CalibrationAuditOptions, FrameInferenceEngine, FrameInferenceOptions,
    FrameInferenceOutput, FrameInferenceTimings, ProviderSelection, SceneSnapshotOutputDestination,
    SceneSnapshotOutputOptions, VideoReplayMode, VideoReplayOptions, VideoReplayReport,
    DEFAULT_VIDEO_MAXIMUM_BYTES, DEFAULT_VIDEO_MAXIMUM_DECODED_BYTES,
    DEFAULT_VIDEO_MAXIMUM_DETAIL_RECORDS, DEFAULT_VIDEO_MAXIMUM_FRAME_COUNT,
    DEFAULT_VIDEO_MAXIMUM_PIXEL_COUNT, DEFAULT_VIDEO_MAXIMUM_PROBE_JSON_BYTES,
    DEFAULT_VIDEO_SUBPROCESS_TIMEOUT_SECONDS,
};
