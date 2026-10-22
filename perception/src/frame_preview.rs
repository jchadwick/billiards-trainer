//! Bounded, Rust-owned still-frame preview for descriptor-bound calibration.
//!
//! The capture mailbox and this module share an `Arc<RgbFrameEnvelope>`, so
//! preview retains the exact budget-accounted RGB allocation used by inference
//! without copying pixels or creating per-client frame queues. Capture only
//! swaps one reference into the slot; JPEG work and all socket I/O happen on
//! bounded preview worker threads.
//!
//! Local protocol v1 is unauthenticated and binds only to a loopback address.
//! A client sends the eight bytes `BTPV0001`. The response is a four-byte
//! big-endian JSON-header length, that many UTF-8 JSON bytes, and exactly the
//! `jpeg_bytes` raw JPEG bytes named by an `ok` header. The response header has
//! `schema_version`, `status`, `descriptor`, `jpeg_bytes`, and `error` fields.
//! Status values are `ok`, `unavailable`, `busy`, `invalid_request`,
//! `oversized`, and `encode_failed`. Non-`ok` responses have a null descriptor,
//! zero JPEG length, and a short error code or null. There is no authentication
//! mechanism; loopback binding is the only network boundary.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use image::codecs::jpeg::JpegEncoder;
use image::ExtendedColorType;
use serde::{Deserialize, Serialize};

use crate::calibration_control::{
    handle_request as handle_calibration_request, write_busy_response, CalibrationControlConfig,
    CALIBRATION_APPLY_REQUEST_V1, CALIBRATION_READ_REQUEST_V1,
};
use crate::calibration_store::CalibrationCaptureContextV1;
use crate::capture::{CaptureTimestampSemantics, RgbFrameEnvelope};
use crate::detection_batch::{BatchTimestamp, ClockKind, TimeUnit, TimestampStage};
use crate::live_config::CaptureSessionCompatibility;

const PREVIEW_REQUEST_V1: &[u8; 8] = b"BTPV0001";
const PREVIEW_PROTOCOL_VERSION: u8 = 1;
const MAXIMUM_PREVIEW_HEADER_BYTES: usize = 1_048_576;
const MAXIMUM_PREVIEW_JPEG_BYTES: usize = 16 * 1_048_576;
const MAXIMUM_PREVIEW_ENCODERS: usize = 4;

fn batch_image_space(image_space: crate::ImageSpace) -> crate::detection_batch::ImageSpace {
    match image_space {
        crate::ImageSpace::Raw => crate::detection_batch::ImageSpace::Raw,
        crate::ImageSpace::Rectified => crate::detection_batch::ImageSpace::Rectified,
    }
}

/// Runtime bounds for one local current-frame preview listener.
///
/// Every value is resolved by the caller; this type intentionally supplies no
/// production defaults. `maximum_jpeg_bytes * maximum_concurrent_encodes` is
/// the strict upper bound for simultaneous JPEG output buffers.
#[derive(Debug, Clone)]
pub struct FramePreviewConfig {
    pub listen_address: SocketAddr,
    pub maximum_header_bytes: usize,
    pub maximum_jpeg_bytes: usize,
    pub maximum_concurrent_encodes: usize,
    pub jpeg_quality: u8,
    pub request_timeout: Duration,
    pub write_timeout: Duration,
    /// Optional Rust-owned calibration control protocol multiplexed on this
    /// same loopback listener. Preview remains usable without the store.
    pub calibration_control: Option<CalibrationControlConfig>,
}

impl FramePreviewConfig {
    pub fn validate(&self) -> Result<(), FramePreviewConfigError> {
        if !self.listen_address.ip().is_loopback() {
            return Err(FramePreviewConfigError::NonLoopbackAddress);
        }
        if !(128..=MAXIMUM_PREVIEW_HEADER_BYTES).contains(&self.maximum_header_bytes) {
            return Err(FramePreviewConfigError::InvalidMaximumHeaderBytes);
        }
        if !(1..=MAXIMUM_PREVIEW_JPEG_BYTES).contains(&self.maximum_jpeg_bytes) {
            return Err(FramePreviewConfigError::InvalidMaximumJpegBytes);
        }
        if !(1..=MAXIMUM_PREVIEW_ENCODERS).contains(&self.maximum_concurrent_encodes) {
            return Err(FramePreviewConfigError::InvalidConcurrency);
        }
        if self
            .maximum_jpeg_bytes
            .checked_mul(self.maximum_concurrent_encodes)
            .is_none()
        {
            return Err(FramePreviewConfigError::AggregateJpegBudgetOverflow);
        }
        if !(1..=100).contains(&self.jpeg_quality) {
            return Err(FramePreviewConfigError::InvalidJpegQuality);
        }
        if self.request_timeout.is_zero() {
            return Err(FramePreviewConfigError::ZeroRequestTimeout);
        }
        if self.write_timeout.is_zero() {
            return Err(FramePreviewConfigError::ZeroWriteTimeout);
        }
        if let Some(control) = &self.calibration_control {
            control
                .validate()
                .map_err(|_| FramePreviewConfigError::InvalidCalibrationControl)?;
            if self.maximum_header_bytes < 4_096 {
                return Err(FramePreviewConfigError::InvalidCalibrationControl);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePreviewConfigError {
    NonLoopbackAddress,
    InvalidMaximumHeaderBytes,
    InvalidMaximumJpegBytes,
    InvalidConcurrency,
    AggregateJpegBudgetOverflow,
    InvalidJpegQuality,
    ZeroRequestTimeout,
    ZeroWriteTimeout,
    InvalidCalibrationControl,
}

impl Display for FramePreviewConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonLoopbackAddress => formatter.write_str("frame preview must bind to loopback"),
            Self::InvalidMaximumHeaderBytes => {
                formatter.write_str("maximum preview header bytes must be between 128 and 1048576")
            }
            Self::InvalidMaximumJpegBytes => {
                formatter.write_str("maximum JPEG bytes must be between 1 and 16777216")
            }
            Self::InvalidConcurrency => formatter
                .write_str("maximum concurrent frame preview encodes must be between 1 and 4"),
            Self::AggregateJpegBudgetOverflow => formatter.write_str(
                "maximum JPEG bytes multiplied by encode concurrency exceeds addressable memory",
            ),
            Self::InvalidJpegQuality => formatter.write_str("JPEG quality must be in 1..=100"),
            Self::ZeroRequestTimeout => {
                formatter.write_str("frame preview request timeout must be positive")
            }
            Self::ZeroWriteTimeout => {
                formatter.write_str("frame preview write timeout must be positive")
            }
            Self::InvalidCalibrationControl => {
                formatter.write_str("frame preview calibration control configuration is invalid")
            }
        }
    }
}

impl Error for FramePreviewConfigError {}

/// Current frame identity and calibration provenance, bound to the returned
/// JPEG bytes. Source time is preserved exactly; read-completion time remains
/// a separately labeled process-local monotonic timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FramePreviewDescriptorV1 {
    pub schema_version: u8,
    pub engine_instance_id: String,
    pub capture_session_id: String,
    pub source_frame_id: String,
    pub width_px: u32,
    pub height_px: u32,
    pub pixel_format: String,
    pub image_space: crate::detection_batch::ImageSpace,
    pub camera_profile_id: String,
    pub configuration_revision: u64,
    pub calibration_revision: u64,
    pub rectification_revision: u64,
    pub source_timestamp: Option<BatchTimestamp>,
    pub source_timestamp_semantics: CaptureTimestampSemantics,
    pub host_read_completed: PreviewHostTimestampV1,
}

/// Process-relative timestamp for host read completion; it is not exposure
/// time and must never be compared to another process unless its clock domain
/// has been verified.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewHostTimestampV1 {
    pub ticks: String,
    pub unit: TimeUnit,
    pub clock: ClockKind,
    pub domain_id: String,
    pub stage: TimestampStage,
    pub semantics: String,
}

/// A descriptor and its exact budget-accounted RGB frame.
#[derive(Debug)]
pub struct CapturedFramePreview {
    frame: Arc<RgbFrameEnvelope>,
    descriptor: FramePreviewDescriptorV1,
}

impl CapturedFramePreview {
    pub fn descriptor(&self) -> &FramePreviewDescriptorV1 {
        &self.descriptor
    }

    pub fn rgb8(&self) -> &[u8] {
        self.frame.rgb8()
    }
}

#[derive(Debug)]
struct PreviewSessionMetadata {
    engine_instance_id: String,
    capture_session_id: String,
    camera_profile_id: String,
    image_space: crate::ImageSpace,
    configuration_revision: u64,
    calibration_revision: u64,
    rectification_revision: u64,
    width_px: u32,
    height_px: u32,
    timestamp_semantics: CaptureTimestampSemantics,
    maximum_frame_age: Duration,
}

#[derive(Debug, Default)]
struct PreviewSlotState {
    metadata: Option<PreviewSessionMetadata>,
    latest: Option<Arc<RgbFrameEnvelope>>,
}

/// Capacity-one, session-scoped view of the newest captured frame.
#[derive(Debug)]
pub struct LatestCapturedFrameSlot {
    engine_instance_id: String,
    process_clock_origin: Instant,
    calibration_context_lock: Mutex<()>,
    state: Mutex<PreviewSlotState>,
}

impl LatestCapturedFrameSlot {
    /// Create an empty preview slot. It remains unavailable until the capture
    /// worker binds it to a source report that passed request preflight.
    pub fn new(engine_instance_id: impl Into<String>) -> Result<Self, FramePreviewSlotError> {
        let engine_instance_id = engine_instance_id.into();
        validate_identifier(&engine_instance_id, "engine_instance_id")?;
        Ok(Self {
            engine_instance_id,
            process_clock_origin: Instant::now(),
            calibration_context_lock: Mutex::new(()),
            state: Mutex::new(PreviewSlotState::default()),
        })
    }

    /// Bind descriptor fields only after the source report matched the exact
    /// capture request. The mailbox invokes this before its first publication.
    pub(crate) fn bind_compatible_session(
        &self,
        compatibility: &CaptureSessionCompatibility,
    ) -> Result<(), FramePreviewSlotError> {
        let _context_guard = self.lock_calibration_context();
        if self.engine_instance_id.is_empty() {
            return Err(FramePreviewSlotError::InvalidIdentifier {
                field: "engine_instance_id",
            });
        }
        let applied = compatibility.applied_frame_geometry();
        let metadata = PreviewSessionMetadata {
            engine_instance_id: self.engine_instance_id.clone(),
            capture_session_id: compatibility.session_id().to_owned(),
            camera_profile_id: applied.camera_profile_id().to_owned(),
            image_space: applied.image_space(),
            configuration_revision: applied.configuration_revision(),
            calibration_revision: compatibility.expected_settings().calibration_revision(),
            rectification_revision: applied.rectification_revision(),
            width_px: compatibility.output_frame_geometry().width_px(),
            height_px: compatibility.output_frame_geometry().height_px(),
            timestamp_semantics: compatibility.timestamp_semantics().clone(),
            maximum_frame_age: compatibility.expected_settings().maximum_frame_age(),
        };
        if metadata.capture_session_id != compatibility.source_report().session_id() {
            return Err(FramePreviewSlotError::SessionMismatch);
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A new compatible capture session invalidates every frame from the
        // previous session before the new descriptor context becomes visible.
        state.latest.take();
        state.metadata = Some(metadata);
        Ok(())
    }

    /// Retain the newest successfully published capture frame. The same Arc is
    /// also placed into the inference mailbox, so this only adds a reference.
    pub(crate) fn publish(
        &self,
        frame: Arc<RgbFrameEnvelope>,
    ) -> Result<(), FramePreviewSlotError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(metadata) = state.metadata.as_ref() else {
            return Err(FramePreviewSlotError::SessionNotBound);
        };
        if frame.source().session_id != metadata.capture_session_id {
            return Err(FramePreviewSlotError::SessionMismatch);
        }
        if !timestamp_matches_semantics(
            &metadata.timestamp_semantics,
            frame.source().timestamp.as_ref(),
        ) {
            return Err(FramePreviewSlotError::TimestampSemanticsMismatch);
        }
        // Keep the camera path to one bounded slot replacement. Descriptor
        // strings, timestamp cloning, and JPEG work are deferred until an
        // actual preview request arrives.
        state.latest = Some(frame);
        Ok(())
    }

    /// Return an Arc to the current frame only while it remains inside the
    /// source request's freshness bound. Expired pixels are released eagerly.
    pub fn snapshot(&self) -> Option<CapturedFramePreview> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let metadata = state.metadata.as_ref()?;
        let frame = state.latest.as_ref()?;
        if frame.source().session_id != metadata.capture_session_id {
            state.latest.take();
            return None;
        }
        let is_stale = Instant::now().saturating_duration_since(frame.host_read_completed_at())
            > metadata.maximum_frame_age;
        if is_stale {
            state.latest.take();
            return None;
        }

        let frame = Arc::clone(frame);
        let source = frame.source();
        let host_read_ticks = frame
            .host_read_completed_at()
            .saturating_duration_since(self.process_clock_origin)
            .as_nanos()
            .min(u128::from(u64::MAX))
            .to_string();
        let descriptor = FramePreviewDescriptorV1 {
            schema_version: PREVIEW_PROTOCOL_VERSION,
            engine_instance_id: metadata.engine_instance_id.clone(),
            capture_session_id: source.session_id.clone(),
            source_frame_id: source.frame_id.clone(),
            width_px: source.width_px,
            height_px: source.height_px,
            pixel_format: "rgb8".to_owned(),
            image_space: match metadata.image_space {
                crate::ImageSpace::Raw => crate::detection_batch::ImageSpace::Raw,
                crate::ImageSpace::Rectified => crate::detection_batch::ImageSpace::Rectified,
            },
            camera_profile_id: metadata.camera_profile_id.clone(),
            configuration_revision: metadata.configuration_revision,
            calibration_revision: metadata.calibration_revision,
            rectification_revision: metadata.rectification_revision,
            source_timestamp: source.timestamp.clone(),
            source_timestamp_semantics: metadata.timestamp_semantics.clone(),
            host_read_completed: PreviewHostTimestampV1 {
                ticks: host_read_ticks,
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Monotonic,
                domain_id: metadata.engine_instance_id.clone(),
                stage: TimestampStage::CameraReadComplete,
                semantics: "read_completion_only".to_owned(),
            },
        };
        Some(CapturedFramePreview { frame, descriptor })
    }

    /// Remove the image reference on stop, source failure, or session change.
    pub(crate) fn clear(&self) {
        let _context_guard = self.lock_calibration_context();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.latest.take();
        state.metadata.take();
    }

    /// Prevent a session transition during one calibration-store commit.
    /// Frame publication does not acquire this gate.
    pub(crate) fn lock_calibration_context(&self) -> MutexGuard<'_, ()> {
        self.calibration_context_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Check the stable descriptor identity under the caller-held context
    /// gate. The frozen source-frame ID is deliberately not compared with the
    /// current slot: it proves which returned image the operator selected,
    /// while calibration applies to the stable capture context.
    pub(crate) fn calibration_binding_context(
        &self,
        descriptor: &FramePreviewDescriptorV1,
    ) -> Result<CalibrationCaptureContextV1, FramePreviewSlotError> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(metadata) = state.metadata.as_ref() else {
            return Err(FramePreviewSlotError::SessionNotBound);
        };
        if descriptor.schema_version != PREVIEW_PROTOCOL_VERSION
            || descriptor.engine_instance_id != metadata.engine_instance_id
            || descriptor.capture_session_id != metadata.capture_session_id
            || descriptor.source_frame_id.is_empty()
            || descriptor.source_frame_id.len() > 128
            || descriptor.source_frame_id.chars().any(char::is_control)
            || descriptor.width_px != metadata.width_px
            || descriptor.height_px != metadata.height_px
            || descriptor.pixel_format != "rgb8"
            || descriptor.image_space != batch_image_space(metadata.image_space)
            || descriptor.camera_profile_id != metadata.camera_profile_id
            || descriptor.configuration_revision != metadata.configuration_revision
            || descriptor.calibration_revision != metadata.calibration_revision
            || descriptor.rectification_revision != metadata.rectification_revision
            || descriptor.source_timestamp_semantics != metadata.timestamp_semantics
        {
            return Err(FramePreviewSlotError::CalibrationContextMismatch);
        }
        Ok(CalibrationCaptureContextV1 {
            engine_instance_id: metadata.engine_instance_id.clone(),
            capture_session_id: metadata.capture_session_id.clone(),
            source_frame_id: descriptor.source_frame_id.clone(),
            configuration_revision: metadata.configuration_revision,
            camera_profile_id: metadata.camera_profile_id.clone(),
            image_space: metadata.image_space,
            source_width_px: metadata.width_px,
            source_height_px: metadata.height_px,
            calibration_revision: metadata.calibration_revision,
            rectification_revision: metadata.rectification_revision,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FramePreviewSlotError {
    InvalidIdentifier { field: &'static str },
    SessionMismatch,
    SessionNotBound,
    TimestampSemanticsMismatch,
    CalibrationContextMismatch,
}

impl Display for FramePreviewSlotError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIdentifier { field } => write!(formatter, "invalid {field}"),
            Self::SessionMismatch => {
                formatter.write_str("preview frame belongs to another session")
            }
            Self::SessionNotBound => {
                formatter.write_str("preview session has not passed preflight")
            }
            Self::TimestampSemanticsMismatch => formatter
                .write_str("preview frame timestamp differs from the accepted source report"),
            Self::CalibrationContextMismatch => formatter
                .write_str("calibration descriptor differs from the accepted capture context"),
        }
    }
}

impl Error for FramePreviewSlotError {}

/// Running loopback TCP endpoint. The listener is ready as soon as `bind`
/// returns; capture may still be starting, in which case requests return
/// `unavailable` until the first accepted frame is published.
#[must_use = "drop the server to stop and join its listener and workers"]
pub struct FramePreviewServer {
    address: SocketAddr,
    stopping: Arc<AtomicBool>,
    slot: Arc<LatestCapturedFrameSlot>,
    accept_thread: Option<JoinHandle<()>>,
}

/// Machine-readable startup line emitted for the Python supervisor after the
/// loopback listener is bound. Listener readiness does not imply frame
/// availability; callers receive `unavailable` until capture publishes one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FramePreviewReadyV1 {
    pub schema_version: u8,
    pub address: String,
    pub engine_instance_id: String,
    pub capture_session_id: String,
}

impl std::fmt::Debug for FramePreviewServer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FramePreviewServer")
            .field("address", &self.address)
            .field("accept_thread_running", &self.accept_thread.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FramePreviewStatusV1 {
    Ok,
    Unavailable,
    Busy,
    InvalidRequest,
    Oversized,
    EncodeFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FramePreviewResponseHeaderV1 {
    pub schema_version: u8,
    pub status: FramePreviewStatusV1,
    pub descriptor: Option<FramePreviewDescriptorV1>,
    pub jpeg_bytes: u32,
    pub error: Option<String>,
}

impl FramePreviewServer {
    pub fn bind(
        config: FramePreviewConfig,
        slot: Arc<LatestCapturedFrameSlot>,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let listener = TcpListener::bind(config.listen_address)?;
        let address = listener.local_addr()?;
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_slot = Arc::clone(&slot);
        let worker_stopping = Arc::clone(&stopping);
        let accept_thread = thread::Builder::new()
            .name("frame-preview-accept".to_owned())
            .spawn(move || {
                frame_preview_accept_loop(listener, worker_stopping, worker_slot, config);
            })?;
        Ok(Self {
            address,
            stopping,
            slot,
            accept_thread: Some(accept_thread),
        })
    }

    /// Actual bound address, including the port selected when port zero was
    /// requested. The Python supervisor consumes this address from readiness.
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    pub fn readiness_record(
        &self,
        engine_instance_id: impl Into<String>,
        capture_session_id: impl Into<String>,
    ) -> FramePreviewReadyV1 {
        FramePreviewReadyV1 {
            schema_version: PREVIEW_PROTOCOL_VERSION,
            address: self.address.to_string(),
            engine_instance_id: engine_instance_id.into(),
            capture_session_id: capture_session_id.into(),
        }
    }
}

impl Drop for FramePreviewServer {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        // One loopback connection wakes the single dispatcher blocked in
        // `accept`; it sees the stop flag and drops this wake connection.
        let _ = TcpStream::connect(self.address);
        if let Some(handle) = self.accept_thread.take() {
            let _ = handle.join();
        }
        self.slot.clear();
    }
}

fn frame_preview_accept_loop(
    listener: TcpListener,
    stopping: Arc<AtomicBool>,
    slot: Arc<LatestCapturedFrameSlot>,
    config: FramePreviewConfig,
) {
    // The dispatcher owns at most `maximum_concurrent_encodes` request
    // threads. Completed handles are reaped on each new connection, and an
    // additional connection receives `busy` immediately instead of building a
    // kernel-backed queue of handlers or encoders.
    let mut workers: Vec<JoinHandle<()>> = Vec::new();
    while !stopping.load(Ordering::Acquire) {
        let (mut stream, _) = match listener.accept() {
            Ok(connection) => connection,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) if stopping.load(Ordering::Acquire) => break,
            Err(_) => continue,
        };
        if stopping.load(Ordering::Acquire) {
            drop(stream);
            break;
        }
        let mut index = 0;
        while index < workers.len() {
            if workers[index].is_finished() {
                let finished = workers.swap_remove(index);
                let _ = finished.join();
            } else {
                index += 1;
            }
        }
        if workers.len() >= config.maximum_concurrent_encodes {
            // A valid saturated request gets a bounded overload response. Read
            // its fixed-size request first so closing the socket does not
            // discard unread client bytes (which would reset the response on
            // some TCP stacks). A client that stalls here can delay only the
            // preview dispatcher, and only up to the configured request
            // timeout; capture and inference never wait on this path.
            let _ = stream.set_read_timeout(Some(config.request_timeout));
            let mut request = [0; PREVIEW_REQUEST_V1.len()];
            if stream.read_exact(&mut request).is_err() {
                continue;
            }
            if &request == PREVIEW_REQUEST_V1 {
                let _ = stream.set_write_timeout(Some(config.write_timeout));
                let _ = write_header(
                    &mut stream,
                    &response(
                        FramePreviewStatusV1::Busy,
                        None,
                        0,
                        Some("encode_capacity_full"),
                    ),
                    config.maximum_header_bytes,
                );
            } else if &request == CALIBRATION_READ_REQUEST_V1
                || (&request == CALIBRATION_APPLY_REQUEST_V1
                    && discard_saturated_calibration_body(&mut stream, &config))
            {
                let _ = stream.set_write_timeout(Some(config.write_timeout));
                let _ = write_busy_response(&mut stream, config.maximum_header_bytes);
            }
            continue;
        }
        let worker_slot = Arc::clone(&slot);
        let worker_config = config.clone();
        let worker_name = format!("frame-preview-request-{}", workers.len());
        match thread::Builder::new().name(worker_name).spawn(move || {
            let _ = stream.set_read_timeout(Some(worker_config.request_timeout));
            let _ = stream.set_write_timeout(Some(worker_config.write_timeout));
            let _ = handle_multiplexed_connection(stream, &worker_slot, &worker_config);
        }) {
            Ok(worker) => workers.push(worker),
            Err(_) => {
                // Thread creation failed after the stream moved into the
                // closure, so it is dropped with that closure. The dispatcher
                // remains bounded and does not retain work or block capture.
            }
        }
    }
    for worker in workers {
        let _ = worker.join();
    }
}

/// Consume a bounded calibration payload before returning `busy`; otherwise
/// unread request bytes can reset the response when the socket closes.
fn discard_saturated_calibration_body(stream: &mut TcpStream, config: &FramePreviewConfig) -> bool {
    let Some(control) = config.calibration_control.as_ref() else {
        return false;
    };
    let mut length_bytes = [0_u8; 4];
    if stream.read_exact(&mut length_bytes).is_err() {
        return false;
    }
    let mut remaining = u32::from_be_bytes(length_bytes) as usize;
    if remaining == 0 || remaining > control.maximum_request_bytes {
        return false;
    }
    let mut scratch = [0_u8; 4096];
    while remaining > 0 {
        let chunk_length = remaining.min(scratch.len());
        if stream.read_exact(&mut scratch[..chunk_length]).is_err() {
            return false;
        }
        remaining -= chunk_length;
    }
    true
}

fn handle_multiplexed_connection(
    mut stream: TcpStream,
    slot: &LatestCapturedFrameSlot,
    config: &FramePreviewConfig,
) -> io::Result<()> {
    let mut request = [0; PREVIEW_REQUEST_V1.len()];
    stream.read_exact(&mut request)?;
    if &request == PREVIEW_REQUEST_V1 {
        return handle_frame_preview(stream, slot, config);
    }
    if &request == CALIBRATION_READ_REQUEST_V1 || &request == CALIBRATION_APPLY_REQUEST_V1 {
        return handle_calibration_request(
            stream,
            request,
            slot,
            config.calibration_control.as_ref(),
            config.maximum_header_bytes,
        );
    }
    write_header(
        &mut stream,
        &response(
            FramePreviewStatusV1::InvalidRequest,
            None,
            0,
            Some("bad_request"),
        ),
        config.maximum_header_bytes,
    )
}

fn handle_frame_preview(
    mut stream: TcpStream,
    slot: &LatestCapturedFrameSlot,
    config: &FramePreviewConfig,
) -> io::Result<()> {
    let Some(preview) = slot.snapshot() else {
        return write_header(
            &mut stream,
            &response(
                FramePreviewStatusV1::Unavailable,
                None,
                0,
                Some("no_current_frame"),
            ),
            config.maximum_header_bytes,
        );
    };
    let jpeg = match encode_jpeg(&preview, config) {
        Ok(jpeg) => jpeg,
        Err(JpegEncodeError::OutputLimitExceeded) => {
            return write_header(
                &mut stream,
                &response(
                    FramePreviewStatusV1::Oversized,
                    None,
                    0,
                    Some("jpeg_limit_exceeded"),
                ),
                config.maximum_header_bytes,
            )
        }
        Err(JpegEncodeError::Encoder) | Err(JpegEncodeError::Allocation) => {
            return write_header(
                &mut stream,
                &response(
                    FramePreviewStatusV1::EncodeFailed,
                    None,
                    0,
                    Some("jpeg_encode_failed"),
                ),
                config.maximum_header_bytes,
            )
        }
    };
    let jpeg_length = u32::try_from(jpeg.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "JPEG length exceeds protocol v1",
        )
    })?;
    write_header(
        &mut stream,
        &response(
            FramePreviewStatusV1::Ok,
            Some(preview.descriptor().clone()),
            jpeg_length,
            None,
        ),
        config.maximum_header_bytes,
    )?;
    stream.write_all(&jpeg)
}

fn response(
    status: FramePreviewStatusV1,
    descriptor: Option<FramePreviewDescriptorV1>,
    jpeg_bytes: u32,
    error: Option<&str>,
) -> FramePreviewResponseHeaderV1 {
    FramePreviewResponseHeaderV1 {
        schema_version: PREVIEW_PROTOCOL_VERSION,
        status,
        descriptor,
        jpeg_bytes,
        error: error.map(str::to_owned),
    }
}

fn write_header(
    stream: &mut TcpStream,
    header: &FramePreviewResponseHeaderV1,
    maximum_header_bytes: usize,
) -> io::Result<()> {
    let bytes = serde_json::to_vec(header)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if bytes.len() > maximum_header_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "preview header exceeds its configured byte limit",
        ));
    }
    let length = u32::try_from(bytes.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "preview header too large"))?;
    stream.write_all(&length.to_be_bytes())?;
    stream.write_all(&bytes)
}

#[derive(Debug)]
enum JpegEncodeError {
    OutputLimitExceeded,
    Allocation,
    Encoder,
}

struct BoundedJpegOutput {
    bytes: Vec<u8>,
    maximum_bytes: usize,
    exceeded: bool,
}

impl BoundedJpegOutput {
    fn new(maximum_bytes: usize) -> Result<Self, JpegEncodeError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(maximum_bytes)
            .map_err(|_| JpegEncodeError::Allocation)?;
        if bytes.capacity() > maximum_bytes {
            return Err(JpegEncodeError::Allocation);
        }
        Ok(Self {
            bytes,
            maximum_bytes,
            exceeded: false,
        })
    }
}

impl Write for BoundedJpegOutput {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(new_length) = self.bytes.len().checked_add(buffer.len()) else {
            self.exceeded = true;
            return Err(io::Error::other("JPEG output size overflow"));
        };
        if new_length > self.maximum_bytes {
            self.exceeded = true;
            return Err(io::Error::other("JPEG output limit exceeded"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode_jpeg(
    preview: &CapturedFramePreview,
    config: &FramePreviewConfig,
) -> Result<Vec<u8>, JpegEncodeError> {
    let mut output = BoundedJpegOutput::new(config.maximum_jpeg_bytes)?;
    let encode_result = JpegEncoder::new_with_quality(&mut output, config.jpeg_quality).encode(
        preview.rgb8(),
        preview.descriptor.width_px,
        preview.descriptor.height_px,
        ExtendedColorType::Rgb8,
    );
    if output.exceeded {
        return Err(JpegEncodeError::OutputLimitExceeded);
    }
    encode_result.map_err(|_| JpegEncodeError::Encoder)?;
    Ok(output.bytes)
}

fn timestamp_matches_semantics(
    semantics: &CaptureTimestampSemantics,
    timestamp: Option<&BatchTimestamp>,
) -> bool {
    match (semantics, timestamp) {
        (CaptureTimestampSemantics::ReadCompletionOnly, None) => true,
        (
            CaptureTimestampSemantics::SourceTimestamp {
                unit,
                clock,
                domain_id,
                stage,
            },
            Some(timestamp),
        ) => {
            timestamp.unit == *unit
                && timestamp.clock == *clock
                && timestamp.domain_id == *domain_id
                && timestamp.stage == *stage
        }
        _ => false,
    }
}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), FramePreviewSlotError> {
    if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
        return Err(FramePreviewSlotError::InvalidIdentifier { field });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use serde_json::Value;

    use crate::calibration_control::{
        CalibrationControlConfig, CALIBRATION_APPLY_REQUEST_V1, CALIBRATION_READ_REQUEST_V1,
    };
    use crate::calibration_store::CalibrationStore;
    use crate::capture::{
        CaptureAggregateFrameLimit, CaptureAppliedCameraControls, CaptureAppliedFrameGeometry,
        CaptureCapability, CaptureFrameBudget, CaptureFrameLimit, CaptureFrameRate,
        CapturePixelFormat, CaptureSourceCapabilities, CaptureTimestampSemantics,
        CaptureVideoProfile,
    };
    use crate::detection_batch::{
        BatchTimestamp, ClockKind, ImageSpace as BatchImageSpace, SourceFrame, TimeUnit,
        TimestampStage,
    };
    use crate::live_config::{
        CaptureControlRequirements, CaptureSessionCompatibility, ExpectedCaptureSettings,
    };
    use crate::ImageSpace;

    use super::{
        BoundedJpegOutput, FramePreviewConfig, FramePreviewServer, FramePreviewSlotError,
        LatestCapturedFrameSlot, PREVIEW_REQUEST_V1,
    };

    fn make_slot_and_compatibility(
        maximum_frame_age: Duration,
    ) -> (
        Arc<LatestCapturedFrameSlot>,
        CaptureSessionCompatibility,
        CaptureFrameBudget,
    ) {
        make_slot_and_compatibility_for_session(maximum_frame_age, "session-preview")
    }

    fn make_slot_and_compatibility_for_session(
        maximum_frame_age: Duration,
        session_id: &str,
    ) -> (
        Arc<LatestCapturedFrameSlot>,
        CaptureSessionCompatibility,
        CaptureFrameBudget,
    ) {
        let frame_limit = CaptureFrameLimit::new(1_048_576).unwrap();
        let aggregate_limit = CaptureAggregateFrameLimit::new(4_194_304).unwrap();
        let budget = CaptureFrameBudget::new(aggregate_limit);
        let profile = CaptureVideoProfile::new(
            8,
            6,
            CaptureFrameRate::new(30, 1).unwrap(),
            CapturePixelFormat::new("rgb8").unwrap(),
        )
        .unwrap();
        let expected = ExpectedCaptureSettings::new(
            7,
            "test-device",
            profile.clone(),
            profile.clone(),
            "profile-test",
            ImageSpace::Rectified,
            9,
            11,
            None,
            CaptureTimestampSemantics::SourceTimestamp {
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Sensor,
                domain_id: "sensor-clock".to_owned(),
                stage: TimestampStage::Exposure,
            },
            CaptureControlRequirements::default(),
            maximum_frame_age,
            frame_limit,
            aggregate_limit,
        )
        .unwrap();
        let applied =
            CaptureAppliedFrameGeometry::new(7, "profile-test", ImageSpace::Rectified, 9).unwrap();
        let metadata = super::super::capture::CaptureSourceMetadata::new(
            "test-device",
            Some("Test Camera".to_owned()),
            "test-backend",
            profile.clone(),
            profile,
            CaptureSourceCapabilities::new(
                None,
                CaptureCapability::Unsupported,
                CaptureCapability::Unsupported,
                CaptureCapability::Unsupported,
                CaptureCapability::Supported,
            )
            .unwrap(),
            applied,
            CaptureAppliedCameraControls::unknown(),
            CaptureTimestampSemantics::SourceTimestamp {
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Sensor,
                domain_id: "sensor-clock".to_owned(),
                stage: TimestampStage::Exposure,
            },
        )
        .unwrap();
        let report =
            super::super::capture::CaptureSessionReport::new(session_id.to_owned(), metadata);
        let compatibility = expected.validate_session(&report).unwrap();
        let slot = Arc::new(LatestCapturedFrameSlot::new("engine-preview").unwrap());
        slot.bind_compatible_session(&compatibility).unwrap();
        (slot, compatibility, budget)
    }

    fn source_frame(id: &str, timestamp: u64) -> SourceFrame {
        SourceFrame {
            session_id: "session-preview".to_owned(),
            frame_id: id.to_owned(),
            timestamp: Some(BatchTimestamp {
                ticks: timestamp.to_string(),
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Sensor,
                domain_id: "sensor-clock".to_owned(),
                stage: TimestampStage::Exposure,
            }),
            width_px: 8,
            height_px: 6,
        }
    }

    fn frame(
        budget: &CaptureFrameBudget,
        id: &str,
        timestamp: u64,
        value: u8,
        read_at: Instant,
    ) -> Arc<crate::capture::RgbFrameEnvelope> {
        let limit = CaptureFrameLimit::new(1_048_576).unwrap();
        let mut pixels = budget.allocate_rgb8_buffer(8, 6, limit).unwrap();
        pixels.as_mut_slice().fill(value);
        Arc::new(
            crate::capture::RgbFrameEnvelope::new(
                source_frame(id, timestamp),
                pixels,
                read_at,
                limit,
            )
            .unwrap(),
        )
    }

    fn config(address: SocketAddr) -> FramePreviewConfig {
        FramePreviewConfig {
            listen_address: address,
            maximum_header_bytes: 8192,
            maximum_jpeg_bytes: 64 * 1024,
            maximum_concurrent_encodes: 2,
            jpeg_quality: 85,
            request_timeout: Duration::from_millis(100),
            write_timeout: Duration::from_millis(100),
            calibration_control: None,
        }
    }

    fn request(address: SocketAddr, bytes: &[u8]) -> std::io::Result<(Value, Vec<u8>)> {
        let mut stream = TcpStream::connect(address)?;
        stream.set_read_timeout(Some(Duration::from_millis(300)))?;
        stream.write_all(bytes)?;
        let mut length = [0; 4];
        stream.read_exact(&mut length)?;
        let header_length = u32::from_be_bytes(length) as usize;
        let mut header = vec![0; header_length];
        stream.read_exact(&mut header)?;
        let header: Value = serde_json::from_slice(&header).unwrap();
        let jpeg_length = header["jpeg_bytes"].as_u64().unwrap() as usize;
        let mut jpeg = vec![0; jpeg_length];
        stream.read_exact(&mut jpeg)?;
        Ok((header, jpeg))
    }

    fn control_request(
        address: SocketAddr,
        magic: &[u8; 8],
        body: Option<&[u8]>,
    ) -> std::io::Result<Value> {
        let mut stream = TcpStream::connect(address)?;
        stream.set_read_timeout(Some(Duration::from_millis(500)))?;
        stream.write_all(magic)?;
        if let Some(body) = body {
            let length = u32::try_from(body.len()).unwrap().to_be_bytes();
            stream.write_all(&length)?;
            stream.write_all(body)?;
        }
        let mut length = [0; 4];
        stream.read_exact(&mut length)?;
        let mut bytes = vec![0; u32::from_be_bytes(length) as usize];
        stream.read_exact(&mut bytes)?;
        Ok(serde_json::from_slice(&bytes).unwrap())
    }

    #[test]
    fn preview_tracks_newest_captured_frame_while_inference_retains_an_older_frame() {
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_secs(2));
        let inference_holds = frame(&budget, "frame-1", 1, 17, Instant::now());
        slot.publish(Arc::clone(&inference_holds)).unwrap();
        let newest = frame(&budget, "frame-2", 2, 231, Instant::now());
        slot.publish(Arc::clone(&newest)).unwrap();

        let current = slot.snapshot().unwrap();
        assert_eq!(current.descriptor().source_frame_id, "frame-2");
        assert_eq!(current.rgb8(), newest.rgb8());
        assert_eq!(inference_holds.source().frame_id, "frame-1");
        assert_eq!(
            current
                .descriptor()
                .source_timestamp
                .as_ref()
                .unwrap()
                .ticks,
            "2"
        );
        assert_eq!(current.descriptor().pixel_format, "rgb8");
        assert_eq!(current.descriptor().image_space, BatchImageSpace::Rectified);
        assert_eq!(current.descriptor().camera_profile_id, "profile-test");
        assert_eq!(current.descriptor().configuration_revision, 7);
        assert_eq!(current.descriptor().calibration_revision, 11);
        assert_eq!(current.descriptor().rectification_revision, 9);
        assert_eq!(
            current.descriptor().host_read_completed.stage,
            TimestampStage::CameraReadComplete
        );
        assert_eq!(
            current.descriptor().host_read_completed.semantics,
            "read_completion_only"
        );
        assert_eq!(budget.stats().current_bytes, 8 * 6 * 3 * 2);

        drop(current);
        drop(newest);
        drop(inference_holds);
        slot.clear();
        assert_eq!(budget.stats().current_bytes, 0);
    }

    #[test]
    fn binding_a_restarted_capture_session_discards_the_previous_session_frame() {
        let (slot, _previous_compatibility, budget) =
            make_slot_and_compatibility(Duration::from_secs(2));
        slot.publish(frame(&budget, "frame-old-session", 1, 19, Instant::now()))
            .unwrap();
        assert!(slot.snapshot().is_some());

        let (_unused_slot, restarted_compatibility, _other_budget) =
            make_slot_and_compatibility_for_session(Duration::from_secs(2), "session-restarted");
        slot.bind_compatible_session(&restarted_compatibility)
            .unwrap();

        assert!(slot.snapshot().is_none());
        assert_eq!(budget.stats().current_bytes, 0);
    }

    #[test]
    fn descriptor_cannot_be_bound_to_a_frame_from_another_session_or_timestamp_contract() {
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_secs(1));
        let wrong = Arc::new({
            let valid = frame(&budget, "wrong-session", 3, 1, Instant::now());
            let (mut source, rgb8, read_at) = Arc::try_unwrap(valid).unwrap().into_parts();
            source.session_id = "other-session".to_owned();
            crate::capture::RgbFrameEnvelope::new(
                source,
                rgb8,
                read_at,
                CaptureFrameLimit::new(1_048_576).unwrap(),
            )
            .unwrap()
        });
        assert_eq!(
            slot.publish(wrong),
            Err(FramePreviewSlotError::SessionMismatch)
        );

        let current = frame(&budget, "wrong-timestamp", 4, 2, Instant::now());
        let (mut source, rgb8, read_at) = Arc::try_unwrap(current).unwrap().into_parts();
        source.timestamp.as_mut().unwrap().domain_id = "wrong-clock".to_owned();
        let mismatch = Arc::new(
            crate::capture::RgbFrameEnvelope::new(
                source,
                rgb8,
                read_at,
                CaptureFrameLimit::new(1_048_576).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(
            slot.publish(mismatch),
            Err(FramePreviewSlotError::TimestampSemanticsMismatch)
        );
    }

    #[test]
    fn stale_current_frame_is_unavailable_and_released_from_the_slot() {
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_millis(1));
        let held = frame(
            &budget,
            "old",
            1,
            7,
            Instant::now() - Duration::from_millis(10),
        );
        slot.publish(Arc::clone(&held)).unwrap();
        assert_eq!(Arc::strong_count(&held), 2);
        assert!(slot.snapshot().is_none());
        assert_eq!(Arc::strong_count(&held), 1);
        assert_eq!(budget.stats().current_bytes, held.rgb8().len());
        drop(held);
        assert_eq!(budget.stats().current_bytes, 0);
    }

    #[test]
    fn bounded_jpeg_writer_rejects_bytes_over_the_configured_limit() {
        let mut output = BoundedJpegOutput::new(4).unwrap();
        output.write_all(&[1, 2, 3, 4]).unwrap();
        assert!(output.write_all(&[5]).is_err());
        assert!(output.exceeded);
        assert_eq!(output.bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn protocol_returns_unavailable_then_descriptor_bound_jpeg_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_secs(2));
        let server = FramePreviewServer::bind(config(address), Arc::clone(&slot)).unwrap();

        let (unavailable, no_jpeg) = request(server.address(), PREVIEW_REQUEST_V1).unwrap();
        assert_eq!(unavailable["status"], "unavailable");
        assert_eq!(unavailable["jpeg_bytes"], 0);
        assert!(no_jpeg.is_empty());

        let newest = frame(&budget, "frame-8", 8, 180, Instant::now());
        slot.publish(newest).unwrap();
        let (response, jpeg) = request(server.address(), PREVIEW_REQUEST_V1).unwrap();
        assert_eq!(response["status"], "ok");
        assert_eq!(response["schema_version"], 1);
        assert_eq!(response["descriptor"]["source_frame_id"], "frame-8");
        assert_eq!(
            response["descriptor"]["capture_session_id"],
            "session-preview"
        );
        assert_eq!(
            response["descriptor"]["engine_instance_id"],
            "engine-preview"
        );
        assert_eq!(response["jpeg_bytes"], jpeg.len());
        assert!(jpeg.starts_with(&[0xff, 0xd8]));
        let decoded = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        assert_eq!(decoded.dimensions(), (8, 6));
        assert_eq!(decoded.get_pixel(0, 0).0, [180, 180, 180]);
    }

    #[test]
    fn calibration_control_checks_stable_context_cas_and_persists_frozen_frame_selection() {
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_secs(3));
        let directory = tempfile::tempdir().unwrap();
        let store = CalibrationStore::new(
            directory.path().join("table-calibration.json"),
            Duration::from_millis(100),
            64 * 1024,
        )
        .unwrap();
        let mut server_config = config("127.0.0.1:0".parse().unwrap());
        server_config.calibration_control = Some(CalibrationControlConfig {
            store: store.clone(),
            maximum_request_bytes: 4096,
        });
        let server = FramePreviewServer::bind(server_config, Arc::clone(&slot)).unwrap();

        let first = frame(&budget, "operator-selected-frame", 1, 33, Instant::now());
        slot.publish(first).unwrap();
        let (preview_header, _) = request(server.address(), PREVIEW_REQUEST_V1).unwrap();
        let descriptor = preview_header["descriptor"].clone();
        assert_eq!(descriptor["source_frame_id"], "operator-selected-frame");

        let later = frame(&budget, "newest-frame", 2, 71, Instant::now());
        slot.publish(later).unwrap();
        let mut submit = serde_json::json!({
            "schema_version": 1,
            "descriptor": descriptor,
            "table_width_mm": 2000.0,
            "table_height_mm": 1000.0,
            "corners_px": [
                {"x_px": 0.0, "y_px": 0.0},
                {"x_px": 7.0, "y_px": 0.0},
                {"x_px": 7.0, "y_px": 5.0},
                {"x_px": 0.0, "y_px": 5.0}
            ],
            "expected_store_revision": null
        });
        let saved = control_request(
            server.address(),
            CALIBRATION_APPLY_REQUEST_V1,
            Some(&serde_json::to_vec(&submit).unwrap()),
        )
        .unwrap();
        assert_eq!(saved["status"], "ok");
        assert_eq!(saved["snapshot"]["revision"], 1);
        assert_eq!(
            saved["snapshot"]["binding_context"]["source_frame_id"],
            "operator-selected-frame"
        );
        assert_eq!(
            slot.snapshot().unwrap().descriptor().source_frame_id,
            "newest-frame"
        );

        // A repeated write with the old store revision is rejected and cannot
        // change the canonical value.
        let stale = control_request(
            server.address(),
            CALIBRATION_APPLY_REQUEST_V1,
            Some(&serde_json::to_vec(&submit).unwrap()),
        )
        .unwrap();
        assert_eq!(stale["status"], "stale_revision");
        assert_eq!(store.load().unwrap().revision, Some(1));

        // Stable capture changes are rejected before the store CAS. The source
        // frame itself is intentionally not required to remain latest.
        submit["descriptor"]["configuration_revision"] = serde_json::json!(8);
        submit["expected_store_revision"] = serde_json::json!(1);
        let mismatch = control_request(
            server.address(),
            CALIBRATION_APPLY_REQUEST_V1,
            Some(&serde_json::to_vec(&submit).unwrap()),
        )
        .unwrap();
        assert_eq!(mismatch["status"], "stale_context");
        assert_eq!(store.load().unwrap().revision, Some(1));

        let read = control_request(server.address(), CALIBRATION_READ_REQUEST_V1, None).unwrap();
        assert_eq!(read["status"], "ok");
        assert_eq!(read["snapshot"]["revision"], 1);
    }

    #[test]
    fn calibration_control_rejects_oversized_body_before_allocation() {
        let (slot, _compatibility, _budget) = make_slot_and_compatibility(Duration::from_secs(2));
        let directory = tempfile::tempdir().unwrap();
        let mut server_config = config("127.0.0.1:0".parse().unwrap());
        server_config.calibration_control = Some(CalibrationControlConfig {
            store: CalibrationStore::new(
                directory.path().join("table-calibration.json"),
                Duration::from_millis(100),
                64 * 1024,
            )
            .unwrap(),
            maximum_request_bytes: 1024,
        });
        let server = FramePreviewServer::bind(server_config, slot).unwrap();
        let mut stream = TcpStream::connect(server.address()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_millis(500)))
            .unwrap();
        stream.write_all(CALIBRATION_APPLY_REQUEST_V1).unwrap();
        stream.write_all(&1025_u32.to_be_bytes()).unwrap();
        let mut response_length = [0; 4];
        stream.read_exact(&mut response_length).unwrap();
        let mut response = vec![0; u32::from_be_bytes(response_length) as usize];
        stream.read_exact(&mut response).unwrap();
        let response: Value = serde_json::from_slice(&response).unwrap();
        assert_eq!(response["status"], "invalid_request");
        assert_eq!(response["error"], "request_size_limit");
    }

    #[test]
    fn invalid_request_timeout_and_server_shutdown_are_bounded() {
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_secs(2));
        let address = "127.0.0.1:0".parse().unwrap();
        let mut short = config(address);
        short.request_timeout = Duration::from_millis(30);
        let server = FramePreviewServer::bind(short, Arc::clone(&slot)).unwrap();

        let (invalid, _) = request(server.address(), b"NOTVALID").unwrap();
        assert_eq!(invalid["status"], "invalid_request");
        slot.publish(frame(&budget, "before-shutdown", 9, 31, Instant::now()))
            .unwrap();
        let mut stalled = TcpStream::connect(server.address()).unwrap();
        stalled
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let start = Instant::now();
        let mut response_length = [0; 4];
        assert!(stalled.read_exact(&mut response_length).is_err());
        assert!(start.elapsed() < Duration::from_secs(1));

        let start = Instant::now();
        let address = server.address();
        drop(server);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(TcpStream::connect(address).is_err());
        assert!(slot.snapshot().is_none());
        drop(stalled);
        assert_eq!(budget.stats().current_bytes, 0);
    }

    #[test]
    fn preview_config_rejects_non_loopback_and_unbounded_or_invalid_values() {
        let mut config = config("0.0.0.0:0".parse().unwrap());
        assert!(config.validate().is_err());
        config.listen_address = "127.0.0.1:0".parse().unwrap();
        config.maximum_header_bytes = 0;
        assert!(config.validate().is_err());
        config.maximum_header_bytes = 8192;
        config.maximum_concurrent_encodes = 0;
        assert!(config.validate().is_err());
        config.maximum_concurrent_encodes = 1;
        config.maximum_jpeg_bytes = 0;
        assert!(config.validate().is_err());
        config.maximum_jpeg_bytes = 64;
        config.jpeg_quality = 0;
        assert!(config.validate().is_err());
        config.jpeg_quality = 100;
        config.request_timeout = Duration::ZERO;
        assert!(config.validate().is_err());
        config.request_timeout = Duration::from_millis(1);
        config.write_timeout = Duration::ZERO;
        assert!(config.validate().is_err());
        config.write_timeout = Duration::from_millis(1);
        config.maximum_jpeg_bytes = u32::MAX as usize + 1;
        assert!(config.validate().is_err());
        config.maximum_jpeg_bytes = 64;
        config.maximum_concurrent_encodes = usize::MAX;
        assert_eq!(
            config.validate(),
            Err(super::FramePreviewConfigError::InvalidConcurrency)
        );
    }

    #[test]
    fn encoded_output_above_limit_returns_oversized_without_jpeg_bytes() {
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_secs(2));
        slot.publish(frame(&budget, "oversized", 1, 90, Instant::now()))
            .unwrap();
        let mut server_config = config("127.0.0.1:0".parse().unwrap());
        server_config.maximum_jpeg_bytes = 1;
        let server = FramePreviewServer::bind(server_config, slot).unwrap();

        let (header, jpeg) = request(server.address(), PREVIEW_REQUEST_V1).unwrap();
        assert_eq!(header["status"], "oversized");
        assert_eq!(header["descriptor"], Value::Null);
        assert_eq!(header["jpeg_bytes"], 0);
        assert_eq!(header["error"], "jpeg_limit_exceeded");
        assert!(jpeg.is_empty());
    }

    #[test]
    fn saturated_preview_worker_returns_busy_without_queueing_more_encoders() {
        let (slot, _compatibility, budget) = make_slot_and_compatibility(Duration::from_secs(3));
        let directory = tempfile::tempdir().unwrap();
        slot.publish(frame(&budget, "parallel", 1, 150, Instant::now()))
            .unwrap();
        let mut server_config = config("127.0.0.1:0".parse().unwrap());
        server_config.maximum_concurrent_encodes = 1;
        server_config.request_timeout = Duration::from_millis(500);
        server_config.calibration_control = Some(CalibrationControlConfig {
            store: CalibrationStore::new(
                directory.path().join("table-calibration.json"),
                Duration::from_millis(100),
                64 * 1024,
            )
            .unwrap(),
            maximum_request_bytes: 4096,
        });
        let server = FramePreviewServer::bind(server_config, slot).unwrap();
        let address = server.address();
        // The first client occupies the sole bounded worker without sending a
        // request. The next valid client must receive a busy response, while
        // no extra worker or frame queue is created.
        let _slow_client = TcpStream::connect(address).unwrap();
        let (busy, jpeg) = request(address, PREVIEW_REQUEST_V1).unwrap();
        assert_eq!(busy["status"], "busy");
        assert_eq!(busy["descriptor"], Value::Null);
        assert_eq!(busy["jpeg_bytes"], 0);
        assert_eq!(busy["error"], "encode_capacity_full");
        assert!(jpeg.is_empty());

        let busy_read = control_request(address, CALIBRATION_READ_REQUEST_V1, None).unwrap();
        assert_eq!(busy_read["status"], "busy");

        let busy_apply =
            control_request(address, CALIBRATION_APPLY_REQUEST_V1, Some(b"{}")).unwrap();
        assert_eq!(busy_apply["status"], "busy");
    }
}
