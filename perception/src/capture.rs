//! Device-independent, owned camera-frame boundary and most-recent-offer handoff.
//!
//! This module defines the seam that a platform-specific capture adapter can
//! implement without selecting a camera library or changing inference. It does
//! not open hardware or select a provider. `capture_pipeline` and
//! `LiveCaptureRuntime` can consume these envelopes from an injected source,
//! but no qualified platform camera adapter or recorded-replay launcher is
//! wired to this capture boundary yet.
//! Source timestamps keep the clock and event semantics supplied by capture;
//! the host's read-completion [`Instant`] is carried separately and must not be
//! substituted for exposure or sensor time.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::ops::Deref;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::detection_batch::{
    BatchTimestamp, ClockKind, SourceFrame, TimeUnit, TimestampStage, MAX_IDENTIFIER_CHARACTERS,
};
use crate::frame_preview::{FramePreviewSlotError, LatestCapturedFrameSlot};
use crate::live_config::{
    CapturePreflightError, CaptureSessionCompatibility, ExpectedCaptureSettings,
};
use crate::ImageSpace;

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const RGB_CHANNELS: usize = 3;

/// Helper-process capture durations measured only during an opt-in live audit.
/// These are elapsed durations in the helper's own process; they are never
/// compared with a timestamp from the parent process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureHelperTimingNs {
    pub appsink_pull_wait_ns: u64,
    pub rgb_buffer_allocation_ns: u64,
    pub rgb_mapping_copy_ns: u64,
}

/// Capture-stage timings carried with a frame for optional audit export.
/// `None` means that the source or stage did not provide a measurement.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CaptureFrameTiming {
    pub helper: Option<CaptureHelperTimingNs>,
    pub parent_framed_record_read_ns: Option<u64>,
}

#[cfg(test)]
const TEST_FRAME_LIMIT: CaptureFrameLimit = CaptureFrameLimit {
    maximum_rgb8_bytes: 6,
};

#[cfg(test)]
const TEST_AGGREGATE_FRAME_LIMIT: CaptureAggregateFrameLimit = CaptureAggregateFrameLimit {
    maximum_rgb8_bytes: 1_048_576,
};

/// Configured upper bound for one owned RGB8 frame.
///
/// Source adapters receive this value so they can reject oversized dimensions
/// before allocating or converting the application-owned pixel buffer. The
/// envelope and mailbox repeat the check to defend against alternate adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureFrameLimit {
    maximum_rgb8_bytes: usize,
}

impl CaptureFrameLimit {
    /// Construct a non-zero RGB8 byte limit resolved by runtime configuration.
    pub fn new(maximum_rgb8_bytes: usize) -> Result<Self, CaptureFrameLimitError> {
        if maximum_rgb8_bytes == 0 {
            return Err(CaptureFrameLimitError::Zero);
        }
        Ok(Self { maximum_rgb8_bytes })
    }

    /// Maximum size of one tightly packed RGB8 frame, in bytes.
    pub fn maximum_rgb8_bytes(self) -> usize {
        self.maximum_rgb8_bytes
    }
}

/// Configured upper bound for all application-owned RGB8 frame capacities.
///
/// This value must come from runtime configuration. The Rust capture boundary
/// deliberately has no production default because choosing a memory budget is
/// an application/deployment decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureAggregateFrameLimit {
    maximum_rgb8_bytes: usize,
}

impl CaptureAggregateFrameLimit {
    /// Construct a non-zero aggregate limit resolved by runtime configuration.
    pub fn new(maximum_rgb8_bytes: usize) -> Result<Self, CaptureFrameLimitError> {
        if maximum_rgb8_bytes == 0 {
            return Err(CaptureFrameLimitError::Zero);
        }
        Ok(Self { maximum_rgb8_bytes })
    }

    /// Maximum combined capacity of all application-owned RGB8 frame buffers.
    pub fn maximum_rgb8_bytes(self) -> usize {
        self.maximum_rgb8_bytes
    }
}

/// Current aggregate RGB8 application-frame memory accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CaptureFrameBudgetStats {
    /// Bytes currently reserved by live budgeted buffers.
    pub current_bytes: usize,
    /// Configured total capacity available to all frames in this process.
    pub configured_limit_bytes: usize,
    /// Largest observed `current_bytes` value.
    pub high_water_mark_bytes: usize,
    /// Allocation attempts rejected by aggregate-budget or capacity checks.
    pub rejected_allocations: u64,
}

#[derive(Debug)]
struct CaptureFrameBudgetInner {
    configured_limit_bytes: usize,
    stats: Mutex<CaptureFrameBudgetStats>,
}

/// Shared allocator for application-owned RGB8 frame buffers.
///
/// Allocations reserve their requested length before obtaining a Vec, then
/// adjust the reservation to the Vec's actual capacity. The resulting buffer
/// keeps that reservation alive until it is dropped. A coordinator passes its
/// process-shared instance to each source read; adapters should use this
/// allocator rather than creating an untracked pixel Vec.
#[derive(Debug, Clone)]
pub struct CaptureFrameBudget {
    inner: Arc<CaptureFrameBudgetInner>,
}

impl CaptureFrameBudget {
    /// Create a budget for standalone use or tests.
    ///
    /// Public coordinators use a single process-wide instance initialized by
    /// `CaptureRuntimeCoordinator::new`; independently constructed budgets
    /// cannot publish into that coordinator's mailbox.
    pub fn new(limit: CaptureAggregateFrameLimit) -> Self {
        Self {
            inner: Arc::new(CaptureFrameBudgetInner {
                configured_limit_bytes: limit.maximum_rgb8_bytes,
                stats: Mutex::new(CaptureFrameBudgetStats {
                    configured_limit_bytes: limit.maximum_rgb8_bytes,
                    ..CaptureFrameBudgetStats::default()
                }),
            }),
        }
    }

    /// Allocate a zero-filled buffer after validating its dimensions and
    /// reserving aggregate capacity. Source adapters can then fill it in place.
    pub fn allocate_rgb8_buffer(
        &self,
        width_px: u32,
        height_px: u32,
        frame_limit: CaptureFrameLimit,
    ) -> Result<BudgetedRgb8Buffer, CaptureFrameAllocationError> {
        let required = RgbFrameEnvelope::required_rgb8_len(width_px, height_px, frame_limit)
            .map_err(CaptureFrameAllocationError::InvalidFrame)?;
        let mut reservation = self
            .reserve(required)
            .map_err(CaptureFrameAllocationError::BudgetExhausted)?;
        let mut pixels = Vec::new();
        if pixels.try_reserve_exact(required).is_err() {
            self.record_rejected_allocation();
            return Err(CaptureFrameAllocationError::AllocationFailed {
                requested_bytes: required,
            });
        }
        let capacity = pixels.capacity();
        if capacity > frame_limit.maximum_rgb8_bytes {
            self.record_rejected_allocation();
            return Err(CaptureFrameAllocationError::FrameCapacityExceedsLimit {
                capacity_bytes: capacity,
                maximum_bytes: frame_limit.maximum_rgb8_bytes,
            });
        }
        reservation
            .set_capacity(capacity)
            .map_err(CaptureFrameAllocationError::BudgetExhausted)?;
        pixels.resize(required, 0);
        Ok(BudgetedRgb8Buffer {
            pixels,
            reservation,
        })
    }

    /// Snapshot aggregate allocation accounting.
    pub fn stats(&self) -> CaptureFrameBudgetStats {
        *self
            .inner
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn reserve(&self, bytes: usize) -> Result<BudgetReservation, CaptureBudgetExhaustion> {
        let mut stats = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let available_bytes = self
            .inner
            .configured_limit_bytes
            .saturating_sub(stats.current_bytes);
        if bytes > available_bytes {
            stats.rejected_allocations = stats.rejected_allocations.saturating_add(1);
            return Err(CaptureBudgetExhaustion {
                requested_bytes: bytes,
                current_bytes: stats.current_bytes,
                available_bytes,
                configured_limit_bytes: self.inner.configured_limit_bytes,
            });
        }
        stats.current_bytes += bytes;
        stats.high_water_mark_bytes = stats.high_water_mark_bytes.max(stats.current_bytes);
        Ok(BudgetReservation {
            inner: Arc::clone(&self.inner),
            bytes,
        })
    }

    fn record_rejected_allocation(&self) {
        let mut stats = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        stats.rejected_allocations = stats.rejected_allocations.saturating_add(1);
    }

    fn owns(&self, reservation: &BudgetReservation) -> bool {
        Arc::ptr_eq(&self.inner, &reservation.inner)
    }
}

#[derive(Debug)]
struct BudgetReservation {
    inner: Arc<CaptureFrameBudgetInner>,
    bytes: usize,
}

impl BudgetReservation {
    fn set_capacity(&mut self, capacity_bytes: usize) -> Result<(), CaptureBudgetExhaustion> {
        let mut stats = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if capacity_bytes > self.bytes {
            let additional_bytes = capacity_bytes - self.bytes;
            let available_bytes = self
                .inner
                .configured_limit_bytes
                .saturating_sub(stats.current_bytes);
            if additional_bytes > available_bytes {
                stats.rejected_allocations = stats.rejected_allocations.saturating_add(1);
                return Err(CaptureBudgetExhaustion {
                    requested_bytes: additional_bytes,
                    current_bytes: stats.current_bytes,
                    available_bytes,
                    configured_limit_bytes: self.inner.configured_limit_bytes,
                });
            }
            stats.current_bytes += additional_bytes;
        } else if capacity_bytes < self.bytes {
            stats.current_bytes = stats
                .current_bytes
                .saturating_sub(self.bytes - capacity_bytes);
        }
        self.bytes = capacity_bytes;
        stats.high_water_mark_bytes = stats.high_water_mark_bytes.max(stats.current_bytes);
        Ok(())
    }
}

impl Drop for BudgetReservation {
    fn drop(&mut self) {
        let mut stats = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(stats.current_bytes >= self.bytes);
        stats.current_bytes = stats.current_bytes.saturating_sub(self.bytes);
    }
}

/// RGB8 Vec whose actual capacity is reserved against a `CaptureFrameBudget`.
///
/// Its Vec is private so callers cannot extract an unbudgeted allocation.
/// Mutable access is limited to the initialized image bytes and cannot grow the
/// capacity.
#[derive(Debug)]
pub struct BudgetedRgb8Buffer {
    pixels: Vec<u8>,
    reservation: BudgetReservation,
}

impl BudgetedRgb8Buffer {
    /// Borrow all initialized image bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.pixels
    }

    /// Mutate initialized image bytes without changing the reserved capacity.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Number of initialized RGB bytes.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Whether the image buffer is empty. Valid RGB frames are never empty.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Actual Vec capacity charged against the aggregate budget.
    pub fn capacity(&self) -> usize {
        self.pixels.capacity()
    }
}

impl PartialEq for BudgetedRgb8Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.pixels == other.pixels
    }
}

impl Eq for BudgetedRgb8Buffer {}

impl std::ops::Deref for BudgetedRgb8Buffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for BudgetedRgb8Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Details of an aggregate frame-memory allocation rejected by the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureBudgetExhaustion {
    pub requested_bytes: usize,
    pub current_bytes: usize,
    pub available_bytes: usize,
    pub configured_limit_bytes: usize,
}

/// Why an RGB8 allocation could not be created.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureFrameAllocationError {
    InvalidFrame(CaptureEnvelopeError),
    BudgetExhausted(CaptureBudgetExhaustion),
    AllocationFailed {
        requested_bytes: usize,
    },
    FrameCapacityExceedsLimit {
        capacity_bytes: usize,
        maximum_bytes: usize,
    },
}

impl Display for CaptureFrameAllocationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFrame(error) => Display::fmt(error, formatter),
            Self::BudgetExhausted(exhaustion) => write!(
                formatter,
                "RGB frame allocation needs {} additional bytes; {} of {} bytes are currently available",
                exhaustion.requested_bytes,
                exhaustion.available_bytes,
                exhaustion.configured_limit_bytes
            ),
            Self::AllocationFailed { requested_bytes } => write!(
                formatter,
                "unable to allocate the required {requested_bytes}-byte RGB frame buffer"
            ),
            Self::FrameCapacityExceedsLimit {
                capacity_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "RGB frame allocation capacity is {capacity_bytes} bytes; per-frame maximum is {maximum_bytes}"
            ),
        }
    }
}

impl Error for CaptureFrameAllocationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidFrame(error) => Some(error),
            Self::BudgetExhausted(_)
            | Self::AllocationFailed { .. }
            | Self::FrameCapacityExceedsLimit { .. } => None,
        }
    }
}

/// Nonfatal reason a source discarded an observed frame and continued reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureFrameDropReason {
    BudgetExhausted(CaptureBudgetExhaustion),
}

impl CaptureFrameAllocationError {
    /// Convert budget exhaustion into a nonfatal source-frame drop reason.
    pub fn drop_reason(&self) -> Option<CaptureFrameDropReason> {
        match self {
            Self::BudgetExhausted(exhaustion) => {
                Some(CaptureFrameDropReason::BudgetExhausted(*exhaustion))
            }
            Self::InvalidFrame(_)
            | Self::AllocationFailed { .. }
            | Self::FrameCapacityExceedsLimit { .. } => None,
        }
    }
}

/// Invalid configured capture-frame memory bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureFrameLimitError {
    Zero,
}

impl Display for CaptureFrameLimitError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => formatter.write_str("configured RGB frame byte limit must be positive"),
        }
    }
}

impl Error for CaptureFrameLimitError {}

/// An owned, tightly packed RGB8 image tied to its original source identity.
///
/// `source.timestamp` is preserved exactly as supplied, including `None` and
/// its original clock domain. In particular, [`TimestampStage::SharedMemoryPublication`]
/// remains a publication timestamp and is never relabeled as camera exposure.
/// `host_read_completed_at` is a separate process-local monotonic point useful
/// for measuring time after the host finished obtaining the frame.
#[derive(Debug, PartialEq)]
pub struct RgbFrameEnvelope {
    /// Source session, frame ID, dimensions, and optional source-clock time.
    source: SourceFrame,
    /// Tightly packed RGB pixels in row-major order, exactly three bytes/pixel.
    rgb8: BudgetedRgb8Buffer,
    /// Host monotonic time when the adapter completed reading this frame.
    host_read_completed_at: Instant,
    /// Optional scalar timings from the helper and parent framed read.
    capture_timing: CaptureFrameTiming,
}

impl RgbFrameEnvelope {
    /// Construct an envelope after validating identity, dimensions, and pixels.
    ///
    /// Capture time is optional because some devices cannot provide a
    /// trustworthy source clock. When present, inference timestamps and the
    /// detector's batch-publication timestamp are rejected as source time.
    /// The frame-arrival and shared-memory-publication stages are retained as
    /// their own events rather than being treated as exposure time.
    pub fn new(
        source: SourceFrame,
        rgb8: BudgetedRgb8Buffer,
        host_read_completed_at: Instant,
        frame_limit: CaptureFrameLimit,
    ) -> Result<Self, CaptureEnvelopeError> {
        Self::new_with_capture_timing(
            source,
            rgb8,
            host_read_completed_at,
            CaptureFrameTiming::default(),
            frame_limit,
        )
    }

    /// Construct an owned frame and attach optional scalar capture timings.
    /// Untimed sources use [`CaptureFrameTiming::default`].
    pub fn new_with_capture_timing(
        source: SourceFrame,
        rgb8: BudgetedRgb8Buffer,
        host_read_completed_at: Instant,
        capture_timing: CaptureFrameTiming,
        frame_limit: CaptureFrameLimit,
    ) -> Result<Self, CaptureEnvelopeError> {
        validate_identifier(&source.session_id, "source.session_id")?;
        validate_identifier(&source.frame_id, "source.frame_id")?;
        if source.width_px == 0 || source.height_px == 0 {
            return Err(CaptureEnvelopeError::InvalidDimensions {
                width_px: source.width_px,
                height_px: source.height_px,
            });
        }
        let expected = Self::required_rgb8_len(source.width_px, source.height_px, frame_limit)?;
        if rgb8.len() != expected {
            return Err(CaptureEnvelopeError::PixelLengthMismatch {
                expected,
                actual: rgb8.len(),
            });
        }
        if rgb8.capacity() > frame_limit.maximum_rgb8_bytes {
            return Err(CaptureEnvelopeError::FrameCapacityExceedsLimit {
                capacity: rgb8.capacity(),
                maximum: frame_limit.maximum_rgb8_bytes,
            });
        }
        if let Some(timestamp) = source.timestamp.as_ref() {
            validate_source_timestamp(timestamp)?;
        }

        Ok(Self {
            source,
            rgb8,
            host_read_completed_at,
            capture_timing,
        })
    }

    /// Compute and validate the RGB8 allocation size before a source adapter
    /// allocates or converts pixels. Native buffers inside a camera driver are
    /// outside this application-owned memory bound.
    pub fn required_rgb8_len(
        width_px: u32,
        height_px: u32,
        frame_limit: CaptureFrameLimit,
    ) -> Result<usize, CaptureEnvelopeError> {
        if width_px == 0 || height_px == 0 {
            return Err(CaptureEnvelopeError::InvalidDimensions {
                width_px,
                height_px,
            });
        }
        let required = usize::try_from(width_px)
            .ok()
            .and_then(|width| {
                usize::try_from(height_px)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS))
            .ok_or(CaptureEnvelopeError::ImageSizeOverflow {
                width_px,
                height_px,
            })?;
        if required > frame_limit.maximum_rgb8_bytes {
            return Err(CaptureEnvelopeError::FrameExceedsLimit {
                required,
                maximum: frame_limit.maximum_rgb8_bytes,
            });
        }
        Ok(required)
    }

    /// Source identity and source timestamp, without reinterpreting its clock.
    pub fn source(&self) -> &SourceFrame {
        &self.source
    }

    /// Borrow the tightly packed owned RGB8 pixels.
    pub fn rgb8(&self) -> &[u8] {
        self.rgb8.as_slice()
    }

    /// Host monotonic read-completion time, separate from source time.
    pub fn host_read_completed_at(&self) -> Instant {
        self.host_read_completed_at
    }

    /// Optional local capture durations; helper and parent clocks are kept as
    /// separate elapsed measurements.
    pub fn capture_timing(&self) -> CaptureFrameTiming {
        self.capture_timing
    }

    /// Consume the envelope into its source metadata, pixels, and host time.
    pub fn into_parts(self) -> (SourceFrame, BudgetedRgb8Buffer, Instant) {
        (self.source, self.rgb8, self.host_read_completed_at)
    }
}

/// Validation error for an owned capture frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureEnvelopeError {
    InvalidIdentifier { field: &'static str },
    InvalidDimensions { width_px: u32, height_px: u32 },
    ImageSizeOverflow { width_px: u32, height_px: u32 },
    PixelLengthMismatch { expected: usize, actual: usize },
    FrameExceedsLimit { required: usize, maximum: usize },
    FrameCapacityExceedsLimit { capacity: usize, maximum: usize },
    AllocationFailed { requested: usize },
    InvalidSourceTimestampTicks,
    InvalidSourceTimestampDomain,
    InvalidSourceTimestampStage(TimestampStage),
}

impl Display for CaptureEnvelopeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIdentifier { field } => write!(
                formatter,
                "{field} must be a non-empty identifier of at most {MAX_IDENTIFIER_CHARACTERS} characters without control characters"
            ),
            Self::InvalidDimensions { width_px, height_px } => write!(
                formatter,
                "source image dimensions must be positive, got {width_px}x{height_px}"
            ),
            Self::ImageSizeOverflow { width_px, height_px } => write!(
                formatter,
                "RGB byte length overflows for source dimensions {width_px}x{height_px}"
            ),
            Self::PixelLengthMismatch { expected, actual } => write!(
                formatter,
                "RGB image has {actual} bytes; expected exactly {expected}"
            ),
            Self::FrameExceedsLimit { required, maximum } => write!(
                formatter,
                "RGB frame requires {required} bytes; configured maximum is {maximum}"
            ),
            Self::FrameCapacityExceedsLimit { capacity, maximum } => write!(
                formatter,
                "RGB frame allocation capacity is {capacity} bytes; configured maximum is {maximum}"
            ),
            Self::AllocationFailed { requested } => write!(
                formatter,
                "unable to allocate the required {requested}-byte RGB frame buffer"
            ),
            Self::InvalidSourceTimestampTicks => write!(
                formatter,
                "source timestamp ticks must be an unsigned 64-bit decimal nanosecond count"
            ),
            Self::InvalidSourceTimestampDomain => write!(
                formatter,
                "source timestamp domain_id must be a valid non-empty identifier"
            ),
            Self::InvalidSourceTimestampStage(stage) => write!(
                formatter,
                "{stage:?} is an inference/output timestamp and cannot identify a source frame"
            ),
        }
    }
}

impl Error for CaptureEnvelopeError {}

fn validate_identifier(value: &str, field: &'static str) -> Result<(), CaptureEnvelopeError> {
    if value.is_empty()
        || value.chars().count() > MAX_IDENTIFIER_CHARACTERS
        || value.chars().any(char::is_control)
    {
        return Err(CaptureEnvelopeError::InvalidIdentifier { field });
    }
    Ok(())
}

fn valid_profile_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER_CHARACTERS
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
}

fn validate_source_timestamp(timestamp: &BatchTimestamp) -> Result<(), CaptureEnvelopeError> {
    validate_identifier(&timestamp.domain_id, "source.timestamp.domain_id")
        .map_err(|_| CaptureEnvelopeError::InvalidSourceTimestampDomain)?;
    timestamp
        .ticks
        .parse::<u64>()
        .map_err(|_| CaptureEnvelopeError::InvalidSourceTimestampTicks)?;
    if matches!(
        timestamp.stage,
        TimestampStage::InferenceStarted
            | TimestampStage::InferenceComplete
            | TimestampStage::BatchPublication
    ) {
        return Err(CaptureEnvelopeError::InvalidSourceTimestampStage(
            timestamp.stage,
        ));
    }
    Ok(())
}

/// A thread-safe, capacity-one handoff that retains the most recently offered frame.
///
/// This mailbox is scoped to exactly one source session. Before constructing a
/// mailbox for a new session, the caller must stop and join the producer and
/// consumer that own the old mailbox. A mailbox does not reopen or change its
/// session. Producers may replace pending frames; consumers receive the last
/// accepted frame offered to the mailbox, linearized by the mailbox lock. The
/// mailbox does not sort by source timestamp or reject out-of-order timestamps.
/// Consumers can inspect the counters to quantify replacement loss.
#[derive(Debug, Clone)]
struct LatestFrameMailbox {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    session_id: String,
    frame_limit: CaptureFrameLimit,
    frame_budget: CaptureFrameBudget,
    frame_preview: Option<Arc<LatestCapturedFrameSlot>>,
    state: Mutex<MailboxState>,
    available: Condvar,
}

#[derive(Debug, Default)]
struct MailboxState {
    pending: Option<MailboxFrame>,
    terminal: Option<TerminalState>,
    counters: MailboxCounters,
    waiting_receivers: usize,
    receiver_claimed: bool,
}

/// Avoid a per-frame Arc allocation when preview is disabled. With preview
/// enabled, inference and the preview slot share the one accepted envelope.
#[allow(clippy::large_enum_variant)] // The inline owned variant avoids a disabled-mode heap allocation.
#[derive(Debug)]
enum MailboxFrame {
    Owned(RgbFrameEnvelope),
    Shared(Arc<RgbFrameEnvelope>),
}

impl std::ops::Deref for MailboxFrame {
    type Target = RgbFrameEnvelope;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(frame) => frame,
            Self::Shared(frame) => frame,
        }
    }
}

impl PartialEq for MailboxFrame {
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TerminalState {
    Closed,
    Cancelled,
    Failed(String),
}

#[derive(Debug, Default)]
struct MailboxCounters {
    accepted_frames: u64,
    replaced_frames: u64,
    rejected_frames: u64,
    cancelled_frames: u64,
    max_queue_depth: usize,
}

/// A point-in-time view of mailbox delivery and loss counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MailboxStats {
    /// Number of frames accepted by the mailbox, including replacement frames.
    pub accepted_frames: u64,
    /// Pending frames discarded when a newer frame replaced them.
    pub replaced_frames: u64,
    /// Frames rejected because of session mismatch or a terminal mailbox.
    pub rejected_frames: u64,
    /// Pending frame discarded by cancellation.
    pub cancelled_frames: u64,
    /// Current depth, always either zero or one.
    pub queue_depth: usize,
    /// Maximum observed queue depth, always zero or one.
    pub max_queue_depth: usize,
    /// Receiver calls currently waiting on the mailbox condition variable.
    pub waiting_receivers: usize,
}

/// Why a producer frame was rejected by the mailbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublishError {
    SessionMismatch { expected: String, actual: String },
    ForeignBudget,
    FrameExceedsLimit { actual: usize, maximum: usize },
    FrameCapacityExceedsLimit { capacity: usize, maximum: usize },
    FramePreview(FramePreviewSlotError),
    Terminal(MailboxTerminal),
}

impl Display for PublishError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionMismatch { expected, actual } => write!(
                formatter,
                "frame belongs to session {actual:?}; mailbox is scoped to {expected:?}"
            ),
            Self::ForeignBudget => formatter.write_str(
                "frame allocation is not reserved against this capture runtime's process budget",
            ),
            Self::FrameExceedsLimit { actual, maximum } => write!(
                formatter,
                "RGB frame has {actual} bytes; configured maximum is {maximum}"
            ),
            Self::FrameCapacityExceedsLimit { capacity, maximum } => write!(
                formatter,
                "RGB frame allocation capacity is {capacity} bytes; configured maximum is {maximum}"
            ),
            Self::FramePreview(error) => {
                write!(formatter, "frame preview publication failed: {error}")
            }
            Self::Terminal(reason) => {
                write!(formatter, "mailbox no longer accepts frames: {reason}")
            }
        }
    }
}

impl Error for PublishError {}

/// Terminal state reported after all queued frames have been handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MailboxTerminal {
    Closed,
    Cancelled,
    ProducerFailed(String),
}

impl Display for MailboxTerminal {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => formatter.write_str("closed"),
            Self::Cancelled => formatter.write_str("cancelled"),
            Self::ProducerFailed(message) => write!(formatter, "producer failed: {message}"),
        }
    }
}

/// Outcome of one blocking or timed mailbox receive operation.
#[derive(Debug, PartialEq)]
enum ReceiveOutcome {
    /// The most recent queued frame was received.
    Frame(MailboxFrame),
    /// The deadline elapsed while the mailbox was open and empty.
    TimedOut,
    /// Graceful closure; any final pending frame was already drained.
    Closed,
    /// Cancellation discarded pending work and ended the mailbox.
    Cancelled,
    /// The producer failed; pending work is delivered before this outcome.
    ProducerFailed(String),
}

/// Result from claiming the one consumer slot for a capture session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureReceiverClaimError {
    /// The coordinator has no active capture session.
    NoActiveSession,
    /// A consumer has already claimed this session's receiver slot.
    AlreadyClaimed,
}

impl Display for CaptureReceiverClaimError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActiveSession => formatter.write_str("no active capture session"),
            Self::AlreadyClaimed => {
                formatter.write_str("the active capture session already has a consumer")
            }
        }
    }
}

impl Error for CaptureReceiverClaimError {}

/// Outcome from processing one frame synchronously through the claimed receiver.
///
/// Frame pixels and metadata are borrowed only for the callback's duration. The
/// callback's return value may carry derived inference results, but cannot carry
/// a borrow of the source frame beyond the callback call.
#[derive(Debug, PartialEq)]
pub enum CaptureReceiveOutcome<T> {
    /// The callback ran synchronously for one newly received frame.
    Frame(T),
    /// The deadline elapsed while the session was open and had no pending frame.
    TimedOut,
    /// Graceful closure; any final pending frame was already delivered.
    Closed,
    /// The session was cancelled.
    Cancelled,
    /// The producer failed after any final pending frame was delivered.
    ProducerFailed(String),
}

/// Sole, non-cloneable consumer handle for one capture session.
///
/// Each receive invokes its callback synchronously with a borrowed frame. The
/// mutable receiver borrow serializes callbacks, and the frame cannot be held
/// by the receiver while another frame is received. A slow callback does not
/// block the capture producer; newer offers continue to replace the single
/// pending frame in the capacity-one mailbox.
pub struct CaptureFrameReceiver {
    mailbox: LatestFrameMailbox,
}

impl std::fmt::Debug for CaptureFrameReceiver {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CaptureFrameReceiver")
            .field("session_id", &self.mailbox.session_id())
            .field("mailbox", &self.mailbox.stats())
            .finish()
    }
}

impl CaptureFrameReceiver {
    /// Source session identity associated with every frame delivered here.
    pub fn session_id(&self) -> &str {
        self.mailbox.session_id()
    }

    /// Block until one frame or terminal outcome, invoking `callback` with a
    /// borrowed frame before returning. A slow callback allows the producer to
    /// keep publishing; pending frames coalesce to the latest one.
    pub fn receive_with<T>(
        &mut self,
        callback: impl FnOnce(&RgbFrameEnvelope) -> T,
    ) -> CaptureReceiveOutcome<T> {
        self.receive_with_timeout(Duration::MAX, callback)
    }

    /// Wait up to `timeout` for one frame or terminal outcome, then invoke
    /// `callback` synchronously with the borrowed frame if one is available.
    pub fn receive_with_timeout<T>(
        &mut self,
        timeout: Duration,
        callback: impl FnOnce(&RgbFrameEnvelope) -> T,
    ) -> CaptureReceiveOutcome<T> {
        match self.mailbox.receive_timeout(timeout) {
            ReceiveOutcome::Frame(frame) => CaptureReceiveOutcome::Frame(callback(&frame)),
            ReceiveOutcome::TimedOut => CaptureReceiveOutcome::TimedOut,
            ReceiveOutcome::Closed => CaptureReceiveOutcome::Closed,
            ReceiveOutcome::Cancelled => CaptureReceiveOutcome::Cancelled,
            ReceiveOutcome::ProducerFailed(message) => {
                CaptureReceiveOutcome::ProducerFailed(message)
            }
        }
    }
}

impl LatestFrameMailbox {
    /// Create an empty mailbox for one validated source session identity.
    #[cfg(test)]
    fn with_budget(
        session_id: impl Into<String>,
        frame_limit: CaptureFrameLimit,
        frame_budget: CaptureFrameBudget,
    ) -> Result<Self, CaptureEnvelopeError> {
        Self::with_budget_and_preview(session_id, frame_limit, frame_budget, None)
    }

    fn with_budget_and_preview(
        session_id: impl Into<String>,
        frame_limit: CaptureFrameLimit,
        frame_budget: CaptureFrameBudget,
        frame_preview: Option<Arc<LatestCapturedFrameSlot>>,
    ) -> Result<Self, CaptureEnvelopeError> {
        let session_id = session_id.into();
        validate_identifier(&session_id, "session_id")?;
        Ok(Self {
            inner: Arc::new(Inner {
                session_id,
                frame_limit,
                frame_budget,
                frame_preview,
                state: Mutex::new(MailboxState::default()),
                available: Condvar::new(),
            }),
        })
    }

    #[cfg(test)]
    fn new(
        session_id: impl Into<String>,
        frame_limit: CaptureFrameLimit,
    ) -> Result<Self, CaptureEnvelopeError> {
        Self::with_budget(session_id, frame_limit, test_frame_budget())
    }

    /// Session identity enforced for every published frame.
    /// Capture session to which this readiness result belongs.
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// Offer a validated owned frame, replacing any previously pending frame.
    ///
    /// A rejected frame increments `rejected_frames`; replacing a queued frame
    /// increments `replaced_frames`. In either case queue depth remains at most
    /// one. Accepted frames wake a waiting consumer. Concurrent offers are
    /// ordered by acquisition of the mailbox lock, not by source timestamp.
    fn publish(&self, frame: RgbFrameEnvelope) -> Result<(), PublishError> {
        let mut state = self.lock_state();
        if frame.source.session_id != self.inner.session_id {
            state.counters.rejected_frames = state.counters.rejected_frames.saturating_add(1);
            return Err(PublishError::SessionMismatch {
                expected: self.inner.session_id.clone(),
                actual: frame.source.session_id,
            });
        }
        if !self.inner.frame_budget.owns(&frame.rgb8.reservation) {
            state.counters.rejected_frames = state.counters.rejected_frames.saturating_add(1);
            return Err(PublishError::ForeignBudget);
        }
        if frame.rgb8.len() > self.inner.frame_limit.maximum_rgb8_bytes {
            state.counters.rejected_frames = state.counters.rejected_frames.saturating_add(1);
            return Err(PublishError::FrameExceedsLimit {
                actual: frame.rgb8.len(),
                maximum: self.inner.frame_limit.maximum_rgb8_bytes,
            });
        }
        if frame.rgb8.capacity() > self.inner.frame_limit.maximum_rgb8_bytes {
            state.counters.rejected_frames = state.counters.rejected_frames.saturating_add(1);
            return Err(PublishError::FrameCapacityExceedsLimit {
                capacity: frame.rgb8.capacity(),
                maximum: self.inner.frame_limit.maximum_rgb8_bytes,
            });
        }
        if let Some(terminal) = state.terminal.as_ref().map(TerminalState::as_public) {
            state.counters.rejected_frames = state.counters.rejected_frames.saturating_add(1);
            return Err(PublishError::Terminal(terminal));
        }

        let mailbox_frame = match &self.inner.frame_preview {
            Some(preview) => {
                let frame = Arc::new(frame);
                if let Err(error) = preview.publish(Arc::clone(&frame)) {
                    state.counters.rejected_frames =
                        state.counters.rejected_frames.saturating_add(1);
                    return Err(PublishError::FramePreview(error));
                }
                MailboxFrame::Shared(frame)
            }
            None => MailboxFrame::Owned(frame),
        };
        if state.pending.replace(mailbox_frame).is_some() {
            state.counters.replaced_frames = state.counters.replaced_frames.saturating_add(1);
        }
        state.counters.accepted_frames = state.counters.accepted_frames.saturating_add(1);
        state.counters.max_queue_depth = 1;
        debug_assert!(state.pending.is_some());
        drop(state);
        self.inner.available.notify_one();
        Ok(())
    }

    fn bind_frame_preview(
        &self,
        compatibility: &CaptureSessionCompatibility,
    ) -> Result<(), FramePreviewSlotError> {
        if let Some(preview) = &self.inner.frame_preview {
            preview.bind_compatible_session(compatibility)?;
        }
        Ok(())
    }

    /// Gracefully stop publishing while allowing a pending final frame to drain.
    ///
    /// Returns `true` only when this call establishes the terminal state. Later
    /// terminal operations are idempotent and return `false`.
    fn close(&self) -> bool {
        self.finish(TerminalState::Closed)
    }

    /// Cancel the mailbox and discard its pending frame, if any.
    ///
    /// Returns `true` only when this call establishes the terminal state.
    fn cancel(&self) -> bool {
        let mut state = self.lock_state();
        if state.terminal.is_some() {
            return false;
        }
        if state.pending.take().is_some() {
            state.counters.cancelled_frames = state.counters.cancelled_frames.saturating_add(1);
        }
        state.terminal = Some(TerminalState::Cancelled);
        if let Some(preview) = &self.inner.frame_preview {
            preview.clear();
        }
        drop(state);
        self.inner.available.notify_all();
        true
    }

    /// Report capture-producer failure and wake consumers.
    ///
    /// If a frame is pending, a consumer receives it before the failure. This
    /// preserves the last complete observation while still making the producer
    /// failure explicit. Returns `true` only when this call establishes the
    /// terminal state.
    fn fail(&self, message: impl Into<String>) -> bool {
        self.finish(TerminalState::Failed(message.into()))
    }

    /// Wait indefinitely for a frame or for a terminal mailbox outcome.
    #[cfg(test)]
    fn receive(&self) -> ReceiveOutcome {
        let mut state = self.lock_state();
        loop {
            if let Some(outcome) = take_ready(&mut state) {
                return outcome;
            }
            state.waiting_receivers = state.waiting_receivers.saturating_add(1);
            state = self
                .inner
                .available
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.waiting_receivers = state.waiting_receivers.saturating_sub(1);
        }
    }

    /// Wait up to `timeout` for a frame or terminal mailbox outcome.
    fn receive_timeout(&self, timeout: Duration) -> ReceiveOutcome {
        let started_at = Instant::now();
        let mut state = self.lock_state();
        loop {
            if let Some(outcome) = take_ready(&mut state) {
                return outcome;
            }

            let remaining = timeout.saturating_sub(started_at.elapsed());
            if remaining.is_zero() {
                return ReceiveOutcome::TimedOut;
            }
            state.waiting_receivers = state.waiting_receivers.saturating_add(1);
            let (mut next_state, wait_result) = self
                .inner
                .available
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            next_state.waiting_receivers = next_state.waiting_receivers.saturating_sub(1);
            state = next_state;
            if wait_result.timed_out() && state.pending.is_none() && state.terminal.is_none() {
                return ReceiveOutcome::TimedOut;
            }
        }
    }

    /// Snapshot queue depth, cumulative loss counters, and waiting receivers.
    fn stats(&self) -> MailboxStats {
        let state = self.lock_state();
        MailboxStats {
            accepted_frames: state.counters.accepted_frames,
            replaced_frames: state.counters.replaced_frames,
            rejected_frames: state.counters.rejected_frames,
            cancelled_frames: state.counters.cancelled_frames,
            queue_depth: usize::from(state.pending.is_some()),
            max_queue_depth: state.counters.max_queue_depth,
            waiting_receivers: state.waiting_receivers,
        }
    }

    fn claim_receiver(&self) -> Result<CaptureFrameReceiver, CaptureReceiverClaimError> {
        let mut state = self.lock_state();
        if state.receiver_claimed {
            return Err(CaptureReceiverClaimError::AlreadyClaimed);
        }
        state.receiver_claimed = true;
        drop(state);
        Ok(CaptureFrameReceiver {
            mailbox: self.clone(),
        })
    }

    fn finish(&self, terminal: TerminalState) -> bool {
        let mut state = self.lock_state();
        if state.terminal.is_some() {
            return false;
        }
        state.terminal = Some(terminal);
        if let Some(preview) = &self.inner.frame_preview {
            preview.clear();
        }
        drop(state);
        self.inner.available.notify_all();
        true
    }

    fn lock_state(&self) -> MutexGuard<'_, MailboxState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl TerminalState {
    fn as_public(&self) -> MailboxTerminal {
        match self {
            Self::Closed => MailboxTerminal::Closed,
            Self::Cancelled => MailboxTerminal::Cancelled,
            Self::Failed(message) => MailboxTerminal::ProducerFailed(message.clone()),
        }
    }
}

fn take_ready(state: &mut MailboxState) -> Option<ReceiveOutcome> {
    if let Some(frame) = state.pending.take() {
        return Some(ReceiveOutcome::Frame(frame));
    }
    state.terminal.as_ref().map(|terminal| match terminal {
        TerminalState::Closed => ReceiveOutcome::Closed,
        TerminalState::Cancelled => ReceiveOutcome::Cancelled,
        TerminalState::Failed(message) => ReceiveOutcome::ProducerFailed(message.clone()),
    })
}

/// Thread-safe interruption contract for a source adapter.
///
/// Implementations must be safe to call from the runtime thread while the
/// factory is opening or the source is reading, releasing, or being dropped.
/// The handle must not borrow a native resource that becomes invalid during
/// release. The method should return promptly, be idempotent, and reliably
/// cancel an open/read already in progress or about to begin. If cancellation
/// cannot unblock an operation, `CaptureWorker::stop` can time out and dropping
/// the worker will block while joining it; the worker never detaches the
/// source.
pub trait CaptureInterrupt: Send + Sync + 'static {
    /// Request interruption of the current or next pending source operation.
    fn interrupt(&self);
}

/// Why an unopened source factory could not return an opened source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureOpenError {
    /// Opening failed for a source-specific reason after cleaning up any
    /// partial acquisition.
    Failed(String),
    /// Opening was cancelled by the worker's interrupt handle. The factory
    /// must have cleaned up any partial acquisition before returning.
    Interrupted,
}

/// Rational capture frame rate, expressed as frames per `seconds_denominator`.
///
/// A rational keeps the source's negotiated rate exact (for example,
/// `30000/1001`) and avoids rounding device metadata through floating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CaptureFrameRate {
    frames_numerator: u32,
    seconds_denominator: u32,
}

impl CaptureFrameRate {
    /// Construct a positive rational frame rate.
    pub fn new(
        frames_numerator: u32,
        seconds_denominator: u32,
    ) -> Result<Self, CaptureMetadataError> {
        if frames_numerator == 0 {
            return Err(CaptureMetadataError::ZeroFrameRateNumerator);
        }
        if seconds_denominator == 0 {
            return Err(CaptureMetadataError::ZeroFrameRateDenominator);
        }
        Ok(Self {
            frames_numerator,
            seconds_denominator,
        })
    }

    /// Number of frames in the rational interval.
    pub fn frames_numerator(self) -> u32 {
        self.frames_numerator
    }

    /// Number of seconds in the rational interval.
    pub fn seconds_denominator(self) -> u32 {
        self.seconds_denominator
    }
}

/// Pixel-format identifier reported by a source adapter.
///
/// This is an adapter-neutral code such as a FourCC or a documented format
/// name. It deliberately does not use a camera library's format enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct CapturePixelFormat(String);

impl CapturePixelFormat {
    /// Construct a non-empty, bounded, ASCII pixel-format identifier.
    pub fn new(identifier: impl Into<String>) -> Result<Self, CaptureMetadataError> {
        let identifier = identifier.into();
        if identifier.is_empty()
            || identifier.chars().count() > MAX_IDENTIFIER_CHARACTERS
            || !identifier.is_ascii()
            || !identifier
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        {
            return Err(CaptureMetadataError::InvalidPixelFormat(identifier));
        }
        Ok(Self(identifier))
    }

    /// Source-neutral pixel-format code.
    pub fn identifier(&self) -> &str {
        &self.0
    }
}

/// One requested or negotiated camera capture mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CaptureVideoProfile {
    width_px: u32,
    height_px: u32,
    frame_rate: CaptureFrameRate,
    pixel_format: CapturePixelFormat,
}

impl CaptureVideoProfile {
    /// Construct a capture mode whose dimensions and rate are all non-zero.
    pub fn new(
        width_px: u32,
        height_px: u32,
        frame_rate: CaptureFrameRate,
        pixel_format: CapturePixelFormat,
    ) -> Result<Self, CaptureMetadataError> {
        if width_px == 0 {
            return Err(CaptureMetadataError::ZeroWidth);
        }
        if height_px == 0 {
            return Err(CaptureMetadataError::ZeroHeight);
        }
        Ok(Self {
            width_px,
            height_px,
            frame_rate,
            pixel_format,
        })
    }

    /// Requested or negotiated horizontal frame size.
    pub fn width_px(&self) -> u32 {
        self.width_px
    }

    /// Requested or negotiated vertical frame size.
    pub fn height_px(&self) -> u32 {
        self.height_px
    }

    /// Requested or negotiated frame rate.
    pub fn frame_rate(&self) -> CaptureFrameRate {
        self.frame_rate
    }

    /// Requested or negotiated source pixel format.
    pub fn pixel_format(&self) -> &CapturePixelFormat {
        &self.pixel_format
    }
}

/// Whether a source confirmed, denied, or could not determine a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureCapability {
    Supported,
    Unsupported,
    Unknown,
}

/// A source-reported state for one requested camera control.
///
/// `Applied` contains a readback observed after the adapter configured the
/// opened device. The other states deliberately carry no value, so callers
/// cannot accidentally substitute the request for an unavailable readback.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
pub enum CaptureControlReadback<T> {
    Applied(T),
    Unsupported,
    Unknown,
}

/// Source-confirmed exposure behavior, in source-neutral terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureExposureMode {
    Automatic,
    Manual,
}

/// Effective camera controls returned by a source after opening/configuring.
///
/// Numeric values use the same units/ranges as the capture-settings request.
/// An adapter must convert driver-specific units before reporting `Applied`;
/// it must use `Unknown` when no reliable readback is available.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CaptureAppliedCameraControls {
    exposure_mode: CaptureControlReadback<CaptureExposureMode>,
    exposure_value: CaptureControlReadback<f64>,
    gain: CaptureControlReadback<f64>,
    brightness: CaptureControlReadback<f64>,
    contrast: CaptureControlReadback<f64>,
    saturation: CaptureControlReadback<f64>,
    auto_focus: CaptureControlReadback<bool>,
    buffer_size: CaptureControlReadback<u32>,
}

/// Applied image-space identity for one opened source.
///
/// These values identify the configuration/profile, raw-or-rectified space,
/// and rectification revision. Emitted RGB8 dimensions are reported
/// separately as [`CaptureOutputFrameGeometry`]. Table-calibration application
/// remains owned by the scene pipeline and is validated when it binds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CaptureAppliedFrameGeometry {
    configuration_revision: u64,
    camera_profile_id: String,
    image_space: ImageSpace,
    rectification_revision: u64,
}

/// Dimensions of the tightly packed RGB8 frames emitted to inference.
///
/// These dimensions are independent of the negotiated camera profile: an
/// adapter may decode a packed/compressed input or rectify it before emitting
/// an owned RGB8 frame. The pixel format is fixed by `RgbFrameEnvelope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureOutputFrameGeometry {
    width_px: u32,
    height_px: u32,
}

impl CaptureOutputFrameGeometry {
    /// Construct non-zero emitted RGB8 dimensions.
    pub fn new(width_px: u32, height_px: u32) -> Result<Self, CaptureMetadataError> {
        if width_px == 0 {
            return Err(CaptureMetadataError::ZeroOutputWidth);
        }
        if height_px == 0 {
            return Err(CaptureMetadataError::ZeroOutputHeight);
        }
        Ok(Self {
            width_px,
            height_px,
        })
    }

    pub fn width_px(self) -> u32 {
        self.width_px
    }

    pub fn height_px(self) -> u32 {
        self.height_px
    }
}

impl CaptureAppliedFrameGeometry {
    /// Construct the frame-geometry readback reported by the opened adapter.
    pub fn new(
        configuration_revision: u64,
        camera_profile_id: impl Into<String>,
        image_space: ImageSpace,
        rectification_revision: u64,
    ) -> Result<Self, CaptureMetadataError> {
        let camera_profile_id = camera_profile_id.into();
        if !valid_profile_identifier(&camera_profile_id) {
            return Err(CaptureMetadataError::InvalidIdentifier {
                field: "source.camera_profile_id",
            });
        }
        if configuration_revision == 0 || configuration_revision > MAX_SAFE_JSON_INTEGER {
            return Err(CaptureMetadataError::InvalidRevision(
                "source.configuration_revision",
            ));
        }
        if rectification_revision > MAX_SAFE_JSON_INTEGER {
            return Err(CaptureMetadataError::InvalidRevision(
                "source.rectification_revision",
            ));
        }
        Ok(Self {
            configuration_revision,
            camera_profile_id,
            image_space,
            rectification_revision,
        })
    }

    pub fn configuration_revision(&self) -> u64 {
        self.configuration_revision
    }

    pub fn camera_profile_id(&self) -> &str {
        &self.camera_profile_id
    }

    pub fn image_space(&self) -> ImageSpace {
        self.image_space
    }

    pub fn rectification_revision(&self) -> u64 {
        self.rectification_revision
    }
}

impl CaptureAppliedCameraControls {
    /// Construct readbacks and reject values that violate the shared units.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exposure_mode: CaptureControlReadback<CaptureExposureMode>,
        exposure_value: CaptureControlReadback<f64>,
        gain: CaptureControlReadback<f64>,
        brightness: CaptureControlReadback<f64>,
        contrast: CaptureControlReadback<f64>,
        saturation: CaptureControlReadback<f64>,
        auto_focus: CaptureControlReadback<bool>,
        buffer_size: CaptureControlReadback<u32>,
    ) -> Result<Self, CaptureMetadataError> {
        validate_finite_readback("exposure_value", &exposure_value)?;
        validate_nonnegative_readback("gain", &gain)?;
        validate_unit_interval_readback("brightness", &brightness)?;
        validate_unit_interval_readback("contrast", &contrast)?;
        validate_unit_interval_readback("saturation", &saturation)?;
        if matches!(buffer_size, CaptureControlReadback::Applied(0)) {
            return Err(CaptureMetadataError::InvalidAppliedControl("buffer_size"));
        }
        Ok(Self {
            exposure_mode,
            exposure_value,
            gain,
            brightness,
            contrast,
            saturation,
            auto_focus,
            buffer_size,
        })
    }

    /// Explicit all-unknown state for sources with no control readback API.
    pub fn unknown() -> Self {
        Self {
            exposure_mode: CaptureControlReadback::Unknown,
            exposure_value: CaptureControlReadback::Unknown,
            gain: CaptureControlReadback::Unknown,
            brightness: CaptureControlReadback::Unknown,
            contrast: CaptureControlReadback::Unknown,
            saturation: CaptureControlReadback::Unknown,
            auto_focus: CaptureControlReadback::Unknown,
            buffer_size: CaptureControlReadback::Unknown,
        }
    }

    pub fn exposure_mode(&self) -> &CaptureControlReadback<CaptureExposureMode> {
        &self.exposure_mode
    }

    pub fn exposure_value(&self) -> &CaptureControlReadback<f64> {
        &self.exposure_value
    }

    pub fn gain(&self) -> &CaptureControlReadback<f64> {
        &self.gain
    }

    pub fn brightness(&self) -> &CaptureControlReadback<f64> {
        &self.brightness
    }

    pub fn contrast(&self) -> &CaptureControlReadback<f64> {
        &self.contrast
    }

    pub fn saturation(&self) -> &CaptureControlReadback<f64> {
        &self.saturation
    }

    pub fn auto_focus(&self) -> &CaptureControlReadback<bool> {
        &self.auto_focus
    }

    pub fn buffer_size(&self) -> &CaptureControlReadback<u32> {
        &self.buffer_size
    }
}

fn validate_finite_readback(
    name: &'static str,
    value: &CaptureControlReadback<f64>,
) -> Result<(), CaptureMetadataError> {
    if matches!(value, CaptureControlReadback::Applied(actual) if !actual.is_finite()) {
        return Err(CaptureMetadataError::InvalidAppliedControl(name));
    }
    Ok(())
}

fn validate_nonnegative_readback(
    name: &'static str,
    value: &CaptureControlReadback<f64>,
) -> Result<(), CaptureMetadataError> {
    if matches!(value, CaptureControlReadback::Applied(actual) if !actual.is_finite() || *actual < 0.0)
    {
        return Err(CaptureMetadataError::InvalidAppliedControl(name));
    }
    Ok(())
}

fn validate_unit_interval_readback(
    name: &'static str,
    value: &CaptureControlReadback<f64>,
) -> Result<(), CaptureMetadataError> {
    if matches!(value, CaptureControlReadback::Applied(actual) if !actual.is_finite() || !(0.0..=1.0).contains(actual))
    {
        return Err(CaptureMetadataError::InvalidAppliedControl(name));
    }
    Ok(())
}

/// Camera controls and source features reported after the device is opened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CaptureSourceCapabilities {
    supported_pixel_formats: Option<Vec<CapturePixelFormat>>,
    manual_exposure: CaptureCapability,
    automatic_exposure: CaptureCapability,
    gain_control: CaptureCapability,
    source_timestamps: CaptureCapability,
}

impl CaptureSourceCapabilities {
    /// Construct source capabilities.
    ///
    /// `None` for `supported_pixel_formats` means the adapter could not
    /// enumerate them. `Some` must be a non-empty, duplicate-free list.
    pub fn new(
        supported_pixel_formats: Option<Vec<CapturePixelFormat>>,
        manual_exposure: CaptureCapability,
        automatic_exposure: CaptureCapability,
        gain_control: CaptureCapability,
        source_timestamps: CaptureCapability,
    ) -> Result<Self, CaptureMetadataError> {
        if let Some(formats) = &supported_pixel_formats {
            if formats.is_empty() {
                return Err(CaptureMetadataError::EmptySupportedPixelFormats);
            }
            let mut unique_formats = std::collections::HashSet::with_capacity(formats.len());
            for format in formats {
                if !unique_formats.insert(format) {
                    return Err(CaptureMetadataError::DuplicateSupportedPixelFormat(
                        format.identifier().to_owned(),
                    ));
                }
            }
        }
        Ok(Self {
            supported_pixel_formats,
            manual_exposure,
            automatic_exposure,
            gain_control,
            source_timestamps,
        })
    }

    /// Enumerated source formats, or `None` when the adapter cannot enumerate.
    pub fn supported_pixel_formats(&self) -> Option<&[CapturePixelFormat]> {
        self.supported_pixel_formats.as_deref()
    }

    /// Whether manual exposure control is supported by the opened source.
    pub fn manual_exposure(&self) -> CaptureCapability {
        self.manual_exposure
    }

    /// Whether automatic exposure control is supported by the opened source.
    pub fn automatic_exposure(&self) -> CaptureCapability {
        self.automatic_exposure
    }

    /// Whether gain control is supported by the opened source.
    pub fn gain_control(&self) -> CaptureCapability {
        self.gain_control
    }

    /// Whether the source provides timestamps in addition to read completion.
    pub fn source_timestamps(&self) -> CaptureCapability {
        self.source_timestamps
    }
}

/// Validated requested-versus-negotiated profile and device capability data.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CaptureSourceMetadata {
    device_id: String,
    device_name: Option<String>,
    backend_id: String,
    requested_profile: CaptureVideoProfile,
    negotiated_profile: CaptureVideoProfile,
    capabilities: CaptureSourceCapabilities,
    applied_frame_geometry: CaptureAppliedFrameGeometry,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_frame_geometry: Option<CaptureOutputFrameGeometry>,
    applied_controls: CaptureAppliedCameraControls,
    timestamp_semantics: CaptureTimestampSemantics,
}

impl CaptureSourceMetadata {
    /// Construct metadata and confirm that enumerated modes include the one
    /// the device actually negotiated.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_id: impl Into<String>,
        device_name: Option<String>,
        backend_id: impl Into<String>,
        requested_profile: CaptureVideoProfile,
        negotiated_profile: CaptureVideoProfile,
        capabilities: CaptureSourceCapabilities,
        applied_frame_geometry: CaptureAppliedFrameGeometry,
        applied_controls: CaptureAppliedCameraControls,
        timestamp_semantics: CaptureTimestampSemantics,
    ) -> Result<Self, CaptureMetadataError> {
        let device_id = device_id.into();
        let backend_id = backend_id.into();
        if validate_identifier(&device_id, "source.device_id").is_err() {
            return Err(CaptureMetadataError::InvalidIdentifier {
                field: "source.device_id",
            });
        }
        if validate_identifier(&backend_id, "source.backend_id").is_err() {
            return Err(CaptureMetadataError::InvalidIdentifier {
                field: "source.backend_id",
            });
        }
        if device_name
            .as_deref()
            .is_some_and(|value| validate_identifier(value, "source.device_name").is_err())
        {
            return Err(CaptureMetadataError::InvalidIdentifier {
                field: "source.device_name",
            });
        }
        if let CaptureTimestampSemantics::SourceTimestamp {
            domain_id, stage, ..
        } = &timestamp_semantics
        {
            if validate_identifier(domain_id, "source.timestamp.domain_id").is_err() {
                return Err(CaptureMetadataError::InvalidIdentifier {
                    field: "source.timestamp.domain_id",
                });
            }
            if matches!(
                stage,
                TimestampStage::InferenceStarted
                    | TimestampStage::InferenceComplete
                    | TimestampStage::BatchPublication
            ) {
                return Err(CaptureMetadataError::InvalidTimestampStage(*stage));
            }
        }
        let reports_source_timestamp = matches!(
            &timestamp_semantics,
            CaptureTimestampSemantics::SourceTimestamp { .. }
        );
        if reports_source_timestamp
            != (capabilities.source_timestamps() == CaptureCapability::Supported)
        {
            return Err(CaptureMetadataError::TimestampCapabilityMismatch {
                capability: capabilities.source_timestamps(),
                reports_source_timestamp,
            });
        }
        if let Some(supported_formats) = capabilities.supported_pixel_formats() {
            if !supported_formats.contains(negotiated_profile.pixel_format()) {
                return Err(CaptureMetadataError::NegotiatedPixelFormatNotSupported(
                    negotiated_profile.pixel_format().identifier().to_owned(),
                ));
            }
        }
        Ok(Self {
            device_id,
            device_name,
            backend_id,
            requested_profile,
            negotiated_profile,
            capabilities,
            applied_frame_geometry,
            output_frame_geometry: None,
            applied_controls,
            timestamp_semantics,
        })
    }

    /// Source-reported device identity. This is provenance metadata, not a credential.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Source-reported human-readable device name, when available.
    pub fn device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    /// Source adapter or platform backend identity. This is provenance metadata.
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Profile requested from the device adapter.
    pub fn requested_profile(&self) -> &CaptureVideoProfile {
        &self.requested_profile
    }

    /// Profile actually selected by the device after negotiation.
    pub fn negotiated_profile(&self) -> &CaptureVideoProfile {
        &self.negotiated_profile
    }

    /// Capabilities observed for the opened source.
    pub fn capabilities(&self) -> &CaptureSourceCapabilities {
        &self.capabilities
    }

    /// Readbacks reported by the opened source; never reconstructed from the request.
    pub fn applied_controls(&self) -> &CaptureAppliedCameraControls {
        &self.applied_controls
    }

    /// Applied image-space identity, separate from emitted RGB8 dimensions.
    pub fn applied_frame_geometry(&self) -> &CaptureAppliedFrameGeometry {
        &self.applied_frame_geometry
    }

    /// Explicit emitted RGB8 dimensions for capture-settings v2 sessions.
    ///
    /// `None` is retained for v1 reports and means the legacy output geometry
    /// equals the negotiated camera profile dimensions.
    pub fn output_frame_geometry(&self) -> Option<CaptureOutputFrameGeometry> {
        self.output_frame_geometry
    }

    /// Add v2 emitted-frame geometry without changing the v1 constructor or
    /// its serialized report shape.
    pub fn with_output_frame_geometry(mut self, geometry: CaptureOutputFrameGeometry) -> Self {
        self.output_frame_geometry = Some(geometry);
        self
    }

    /// Timestamp point, clock domain, and stage used by source frame times.
    pub fn timestamp_semantics(&self) -> &CaptureTimestampSemantics {
        &self.timestamp_semantics
    }
}

/// Timestamp meaning declared by the opened adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CaptureTimestampSemantics {
    /// Frames carry no source timestamp; the host read-completion `Instant`
    /// remains separate and must not be interpreted as exposure time.
    ReadCompletionOnly,
    /// Source timestamp clock and event semantics required on every frame.
    SourceTimestamp {
        unit: TimeUnit,
        clock: ClockKind,
        domain_id: String,
        stage: TimestampStage,
    },
}

/// Ready metadata associated with the unique capture session that opened it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct CaptureSessionReport {
    session_id: String,
    device_id: String,
    device_name: Option<String>,
    backend_id: String,
    requested_profile: CaptureVideoProfile,
    negotiated_profile: CaptureVideoProfile,
    capabilities: CaptureSourceCapabilities,
    applied_frame_geometry: CaptureAppliedFrameGeometry,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_frame_geometry: Option<CaptureOutputFrameGeometry>,
    applied_controls: CaptureAppliedCameraControls,
    timestamp_semantics: CaptureTimestampSemantics,
}

impl CaptureSessionReport {
    pub(crate) fn new(session_id: String, metadata: CaptureSourceMetadata) -> Self {
        let CaptureSourceMetadata {
            device_id,
            device_name,
            backend_id,
            requested_profile,
            negotiated_profile,
            capabilities,
            applied_frame_geometry,
            output_frame_geometry,
            applied_controls,
            timestamp_semantics,
        } = metadata;
        Self {
            session_id,
            device_id,
            device_name,
            backend_id,
            requested_profile,
            negotiated_profile,
            capabilities,
            applied_frame_geometry,
            output_frame_geometry,
            applied_controls,
            timestamp_semantics,
        }
    }

    /// Unique capture session that opened the source.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Source-reported device identity. This is provenance metadata, not a credential.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Source-reported human-readable device name, when available.
    pub fn device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    /// Source adapter or platform backend identity. This is provenance metadata.
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Profile requested from the device adapter.
    pub fn requested_profile(&self) -> &CaptureVideoProfile {
        &self.requested_profile
    }

    /// Profile actually selected by the device after negotiation.
    pub fn negotiated_profile(&self) -> &CaptureVideoProfile {
        &self.negotiated_profile
    }

    /// Capabilities observed for the opened source.
    pub fn capabilities(&self) -> &CaptureSourceCapabilities {
        &self.capabilities
    }

    /// Readbacks reported by the opened source; never reconstructed from the request.
    pub fn applied_controls(&self) -> &CaptureAppliedCameraControls {
        &self.applied_controls
    }

    /// Applied image-space identity, separate from emitted RGB8 dimensions.
    pub fn applied_frame_geometry(&self) -> &CaptureAppliedFrameGeometry {
        &self.applied_frame_geometry
    }

    /// Explicit emitted RGB8 dimensions for v2; absent in unchanged v1 reports.
    pub fn output_frame_geometry(&self) -> Option<CaptureOutputFrameGeometry> {
        self.output_frame_geometry
    }

    /// Timestamp point, clock domain, and stage used by source frame times.
    pub fn timestamp_semantics(&self) -> &CaptureTimestampSemantics {
        &self.timestamp_semantics
    }
}

/// Invalid requested, negotiated, or capability metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureMetadataError {
    ZeroWidth,
    ZeroHeight,
    ZeroOutputWidth,
    ZeroOutputHeight,
    ZeroFrameRateNumerator,
    ZeroFrameRateDenominator,
    InvalidIdentifier {
        field: &'static str,
    },
    InvalidPixelFormat(String),
    InvalidTimestampStage(TimestampStage),
    TimestampCapabilityMismatch {
        capability: CaptureCapability,
        reports_source_timestamp: bool,
    },
    EmptySupportedPixelFormats,
    DuplicateSupportedPixelFormat(String),
    NegotiatedPixelFormatNotSupported(String),
    InvalidAppliedControl(&'static str),
    InvalidRevision(&'static str),
}

impl Display for CaptureMetadataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroWidth => formatter.write_str("capture profile width must be non-zero"),
            Self::ZeroHeight => formatter.write_str("capture profile height must be non-zero"),
            Self::ZeroOutputWidth => {
                formatter.write_str("emitted RGB8 frame width must be non-zero")
            }
            Self::ZeroOutputHeight => {
                formatter.write_str("emitted RGB8 frame height must be non-zero")
            }
            Self::ZeroFrameRateNumerator => {
                formatter.write_str("capture frame-rate numerator must be non-zero")
            }
            Self::ZeroFrameRateDenominator => {
                formatter.write_str("capture frame-rate denominator must be non-zero")
            }
            Self::InvalidIdentifier { field } => write!(
                formatter,
                "{field} must be a non-empty identifier of at most {MAX_IDENTIFIER_CHARACTERS} characters without control characters"
            ),
            Self::InvalidPixelFormat(format) => write!(
                formatter,
                "capture pixel-format identifier {format:?} is invalid"
            ),
            Self::InvalidTimestampStage(stage) => write!(
                formatter,
                "{stage:?} is an inference/output timestamp and cannot identify a source frame"
            ),
            Self::TimestampCapabilityMismatch {
                capability,
                reports_source_timestamp,
            } => write!(
                formatter,
                "source timestamp capability is {capability:?}, but metadata reports source timestamp data: {reports_source_timestamp}"
            ),
            Self::EmptySupportedPixelFormats => {
                formatter.write_str("enumerated supported pixel formats must not be empty")
            }
            Self::DuplicateSupportedPixelFormat(format) => write!(
                formatter,
                "enumerated supported pixel format {format:?} is duplicated"
            ),
            Self::NegotiatedPixelFormatNotSupported(format) => write!(
                formatter,
                "negotiated pixel format {format:?} is absent from the enumerated supported formats"
            ),
            Self::InvalidAppliedControl(control) => write!(
                formatter,
                "source-reported applied control {control} is outside the capture-settings contract"
            ),
            Self::InvalidRevision(field) => {
                write!(formatter, "source-reported {field} is outside the safe integer range")
            }
        }
    }
}

impl Error for CaptureMetadataError {}

/// Opened source operations, all invoked on one capture worker thread.
pub trait CaptureSource: Send + 'static {
    /// Return the validated settings and capabilities observed after opening.
    ///
    /// This method runs on the same owner thread as open, read, and release.
    /// The coordinator publishes the result only after the factory has
    /// returned this opened source.
    fn metadata(&self) -> CaptureSourceMetadata;

    /// Read a frame, report a nonfatal dropped observation, or report natural end.
    ///
    /// The worker supplies its unique session ID. Implementations must preserve
    /// each source timestamp's clock and stage; they must not replace exposure
    /// time with read-completion, inference, or publication time.
    fn read(
        &mut self,
        session_id: &str,
        frame_limit: CaptureFrameLimit,
        frame_budget: &CaptureFrameBudget,
    ) -> Result<CaptureReadOutcome, String>;

    /// Release the opened device/resource on this same worker thread.
    fn release(&mut self) -> Result<(), String>;
}

/// Result of one source read operation.
#[derive(Debug, PartialEq)]
pub enum CaptureReadOutcome {
    /// A complete frame whose allocation is reserved against `frame_budget`.
    Frame(RgbFrameEnvelope),
    /// The source observed a frame but could not retain it; the worker continues.
    Dropped(CaptureFrameDropReason),
    /// The source reached its natural end.
    End,
}

/// Unopened adapter that creates an owned source only after its runtime gate is held.
pub trait CaptureSourceFactory: Send + 'static {
    /// Return the thread-safe handle that interrupts the source opened by this
    /// factory. This must not open or acquire the device; it is called before
    /// the worker starts so stop can interrupt an in-progress or future open/read.
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt>;

    /// Open and return the source on the capture worker thread.
    ///
    /// If opening fails or is interrupted, the implementation must clean up
    /// every partial acquisition before returning. A panic does not provide
    /// that guarantee, so the owner gate is quarantined.
    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError>;

    /// Open with the worker's session identity when the source protocol needs
    /// to bind its metadata and frames to that session. Existing factories can
    /// keep implementing `open`; the default preserves their behavior.
    fn open_with_session_id(
        self: Box<Self>,
        _session_id: &str,
    ) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        self.open()
    }
}

/// Shared source-ownership gate for one runtime/provider coordinator.
///
/// Every capture start in one runtime must use the same gate. Each test or
/// separate runtime can construct an isolated gate. The gate is acquired before
/// a factory can open a source, remains owned through worker completion, and is
/// permanently quarantined if release cannot be confirmed. A new runtime can
/// create a new gate after the operating system has released the old device.
#[derive(Debug, Clone)]
struct CaptureOwnerGate {
    state: Arc<Mutex<CaptureOwnerGateState>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CaptureOwnerGateState {
    Available,
    Owned { session_id: String },
    Quarantined { session_id: String, reason: String },
}

/// Current ownership status for a runtime's capture source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureOwnerStatus {
    Available,
    Owned { session_id: String },
    Quarantined { session_id: String, reason: String },
}

/// Why a new capture worker could not reserve the runtime's source gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureOwnerGateError {
    AlreadyOwned { session_id: String },
    Quarantined { session_id: String, reason: String },
}

impl Display for CaptureOwnerGateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyOwned { session_id } => write!(
                formatter,
                "capture source is still owned by session {session_id:?}"
            ),
            Self::Quarantined { session_id, reason } => write!(
                formatter,
                "capture source gate is quarantined after session {session_id:?}: {reason}"
            ),
        }
    }
}

impl Error for CaptureOwnerGateError {}

impl CaptureOwnerGate {
    /// Create an available gate for a runtime/provider coordinator.
    fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(CaptureOwnerGateState::Available)),
        }
    }

    /// Snapshot the gate state without changing ownership.
    fn status(&self) -> CaptureOwnerStatus {
        match &*self.lock_state() {
            CaptureOwnerGateState::Available => CaptureOwnerStatus::Available,
            CaptureOwnerGateState::Owned { session_id } => CaptureOwnerStatus::Owned {
                session_id: session_id.clone(),
            },
            CaptureOwnerGateState::Quarantined { session_id, reason } => {
                CaptureOwnerStatus::Quarantined {
                    session_id: session_id.clone(),
                    reason: reason.clone(),
                }
            }
        }
    }

    fn acquire(&self, session_id: String) -> Result<CaptureOwnerLease, CaptureOwnerGateError> {
        let mut state = self.lock_state();
        match &*state {
            CaptureOwnerGateState::Available => {
                *state = CaptureOwnerGateState::Owned {
                    session_id: session_id.clone(),
                };
                Ok(CaptureOwnerLease {
                    gate: self.clone(),
                    session_id,
                    resolved: false,
                })
            }
            CaptureOwnerGateState::Owned { session_id } => {
                Err(CaptureOwnerGateError::AlreadyOwned {
                    session_id: session_id.clone(),
                })
            }
            CaptureOwnerGateState::Quarantined { session_id, reason } => {
                Err(CaptureOwnerGateError::Quarantined {
                    session_id: session_id.clone(),
                    reason: reason.clone(),
                })
            }
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, CaptureOwnerGateState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for CaptureOwnerGate {
    fn default() -> Self {
        Self::new()
    }
}

// The current runtime supports one physical camera source. All public
// coordinators share this process-wide gate, so constructing a second
// coordinator cannot bypass exclusive ownership. A future multi-camera runtime
// must replace this with a configured, source-keyed owner registry.
static PRIMARY_CAPTURE_OWNER_GATE: OnceLock<CaptureOwnerGate> = OnceLock::new();

fn primary_capture_owner_gate() -> CaptureOwnerGate {
    PRIMARY_CAPTURE_OWNER_GATE
        .get_or_init(CaptureOwnerGate::new)
        .clone()
}

struct PrimaryCaptureFrameBudget {
    limit: CaptureAggregateFrameLimit,
    budget: CaptureFrameBudget,
}

static PRIMARY_CAPTURE_FRAME_BUDGET: OnceLock<Mutex<Option<PrimaryCaptureFrameBudget>>> =
    OnceLock::new();

fn primary_capture_frame_budget(
    requested_limit: CaptureAggregateFrameLimit,
) -> Result<CaptureFrameBudget, CaptureRuntimeConfigurationError> {
    let state = PRIMARY_CAPTURE_FRAME_BUDGET.get_or_init(|| Mutex::new(None));
    let mut configured = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match configured.as_ref() {
        Some(existing) if existing.limit != requested_limit => Err(
            CaptureRuntimeConfigurationError::ConflictingAggregateFrameLimit {
                configured_bytes: existing.limit.maximum_rgb8_bytes,
                requested_bytes: requested_limit.maximum_rgb8_bytes,
            },
        ),
        Some(existing) => Ok(existing.budget.clone()),
        None => {
            let budget = CaptureFrameBudget::new(requested_limit);
            *configured = Some(PrimaryCaptureFrameBudget {
                limit: requested_limit,
                budget: budget.clone(),
            });
            Ok(budget)
        }
    }
}

#[cfg(test)]
thread_local! {
    static TEST_THREAD_FRAME_BUDGET: CaptureFrameBudget =
        CaptureFrameBudget::new(TEST_AGGREGATE_FRAME_LIMIT);
}

#[cfg(test)]
fn test_frame_budget() -> CaptureFrameBudget {
    TEST_THREAD_FRAME_BUDGET.with(Clone::clone)
}

#[derive(Debug)]
struct CaptureOwnerLease {
    gate: CaptureOwnerGate,
    session_id: String,
    resolved: bool,
}

impl CaptureOwnerLease {
    fn release_confirmed(&mut self) {
        let mut state = self.gate.lock_state();
        match &*state {
            CaptureOwnerGateState::Owned { session_id } if session_id == &self.session_id => {
                *state = CaptureOwnerGateState::Available;
            }
            _ => {
                *state = CaptureOwnerGateState::Quarantined {
                    session_id: self.session_id.clone(),
                    reason: "owner gate state changed before release confirmation".to_owned(),
                };
            }
        }
        self.resolved = true;
    }

    fn quarantine(&mut self, reason: impl Into<String>) {
        *self.gate.lock_state() = CaptureOwnerGateState::Quarantined {
            session_id: self.session_id.clone(),
            reason: reason.into(),
        };
        self.resolved = true;
    }
}

impl Drop for CaptureOwnerLease {
    fn drop(&mut self) {
        if !self.resolved {
            self.quarantine("capture worker ended without confirmed release");
        }
    }
}

/// Why a worker completed or stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureWorkerEnd {
    NaturalEnd,
    Stopped,
    Failed,
}

/// A failure surfaced by a capture worker or its stop interrupt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureWorkerFailure {
    Open(String),
    OpenInterrupted,
    OpenPanicked(String),
    SourceMetadataPanicked(String),
    PreflightRejected(CapturePreflightError),
    FramePreviewSetup(String),
    FrameProfileMismatch {
        expected_width_px: u32,
        expected_height_px: u32,
        actual_width_px: u32,
        actual_height_px: u32,
    },
    FrameTimestampMismatch {
        expected: CaptureTimestampSemantics,
        actual: Option<CaptureTimestampSemantics>,
    },
    Read(String),
    ReadPanicked(String),
    PublishRejected(PublishError),
    WorkerPanicked(String),
    Release(String),
    ReleasePanicked(String),
    SourceDropPanicked(String),
    InterruptPanicked(String),
}

impl Display for CaptureWorkerFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(message) => write!(formatter, "source open failed: {message}"),
            Self::OpenInterrupted => {
                formatter.write_str("source open was interrupted without a stop request")
            }
            Self::OpenPanicked(message) => write!(formatter, "source open panicked: {message}"),
            Self::SourceMetadataPanicked(message) => {
                write!(formatter, "source metadata panicked: {message}")
            }
            Self::PreflightRejected(error) => Display::fmt(error, formatter),
            Self::FramePreviewSetup(message) => {
                write!(formatter, "frame preview session setup failed: {message}")
            }
            Self::FrameProfileMismatch {
                expected_width_px,
                expected_height_px,
                actual_width_px,
                actual_height_px,
            } => write!(
                formatter,
                "emitted RGB8 frame dimensions {actual_width_px}x{actual_height_px} do not match preflighted output geometry {expected_width_px}x{expected_height_px}"
            ),
            Self::FrameTimestampMismatch { expected, actual } => write!(
                formatter,
                "frame source timestamp semantics {actual:?} do not match session semantics {expected:?}"
            ),
            Self::Read(message) => write!(formatter, "source read failed: {message}"),
            Self::ReadPanicked(message) => write!(formatter, "source read panicked: {message}"),
            Self::PublishRejected(error) => write!(formatter, "frame publication failed: {error}"),
            Self::WorkerPanicked(message) => {
                write!(formatter, "capture worker panicked: {message}")
            }
            Self::Release(message) => write!(formatter, "source release failed: {message}"),
            Self::ReleasePanicked(message) => {
                write!(formatter, "source release panicked: {message}")
            }
            Self::SourceDropPanicked(message) => {
                write!(formatter, "source destructor panicked: {message}")
            }
            Self::InterruptPanicked(message) => {
                write!(formatter, "source interrupt panicked: {message}")
            }
        }
    }
}

/// Whether the worker proved the source safe to replace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureOwnerDisposition {
    Released,
    Quarantined { reason: String },
}

/// Final source-thread result, available only after the owner thread is joined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureWorkerReport {
    pub session_id: String,
    pub end: CaptureWorkerEnd,
    pub frames_published: u64,
    /// Source observations explicitly skipped after aggregate budget exhaustion.
    pub frames_dropped: u64,
    /// Subset of `frames_dropped` caused by aggregate budget exhaustion.
    pub budget_exhausted_frames: u64,
    pub failures: Vec<CaptureWorkerFailure>,
    pub owner_disposition: CaptureOwnerDisposition,
}

/// Result of a bounded wait or stop request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureWorkerWait {
    Completed(CaptureWorkerReport),
    TimedOut,
}

/// Failure to reserve or start a worker before its factory is opened.
#[derive(Debug)]
pub enum CaptureStartError {
    SessionIdentityExhausted,
    WorkerStillRunning {
        session_id: String,
    },
    OwnerGate(CaptureOwnerGateError),
    InvalidSessionIdentity(CaptureEnvelopeError),
    InterruptHandlePanicked(String),
    PreflightFrameLimitMismatch {
        expected_bytes: usize,
        coordinator_bytes: usize,
    },
    PreflightAggregateLimitMismatch {
        expected_bytes: usize,
        coordinator_bytes: usize,
    },
    ThreadSpawn(io::Error),
}

/// Why a coordinator could not use its requested aggregate frame budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureRuntimeConfigurationError {
    ConflictingAggregateFrameLimit {
        configured_bytes: usize,
        requested_bytes: usize,
    },
}

impl Display for CaptureRuntimeConfigurationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConflictingAggregateFrameLimit {
                configured_bytes,
                requested_bytes,
            } => write!(
                formatter,
                "capture process aggregate frame budget is already configured for {configured_bytes} bytes; this coordinator requested {requested_bytes} bytes"
            ),
        }
    }
}

impl Error for CaptureRuntimeConfigurationError {}

impl Display for CaptureStartError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionIdentityExhausted => {
                formatter.write_str("capture session identity counter is exhausted")
            }
            Self::WorkerStillRunning { session_id } => write!(
                formatter,
                "capture session {session_id:?} must finish before a new source can start"
            ),
            Self::OwnerGate(error) => write!(formatter, "capture worker could not start: {error}"),
            Self::InvalidSessionIdentity(error) => {
                write!(
                    formatter,
                    "capture session identity is invalid: {error}"
                )
            }
            Self::InterruptHandlePanicked(message) => {
                write!(formatter, "source interrupt handle panicked: {message}")
            }
            Self::PreflightFrameLimitMismatch {
                expected_bytes,
                coordinator_bytes,
            } => write!(
                formatter,
                "preflight per-frame limit {expected_bytes} bytes does not match coordinator limit {coordinator_bytes} bytes"
            ),
            Self::PreflightAggregateLimitMismatch {
                expected_bytes,
                coordinator_bytes,
            } => write!(
                formatter,
                "preflight aggregate limit {expected_bytes} bytes does not match coordinator limit {coordinator_bytes} bytes"
            ),
            Self::ThreadSpawn(error) => {
                write!(formatter, "unable to start capture worker thread: {error}")
            }
        }
    }
}

impl Error for CaptureStartError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OwnerGate(error) => Some(error),
            Self::InvalidSessionIdentity(error) => Some(error),
            Self::ThreadSpawn(error) => Some(error),
            Self::SessionIdentityExhausted
            | Self::WorkerStillRunning { .. }
            | Self::InterruptHandlePanicked(_)
            | Self::PreflightFrameLimitMismatch { .. }
            | Self::PreflightAggregateLimitMismatch { .. } => None,
        }
    }
}

struct CaptureWorkerInput {
    factory: Box<dyn CaptureSourceFactory>,
    session_id: String,
    frame_limit: CaptureFrameLimit,
    frame_budget: CaptureFrameBudget,
    mailbox: LatestFrameMailbox,
    stop_requested: Arc<AtomicBool>,
    ready_report: Arc<Mutex<Option<CaptureSessionReport>>>,
    preflight: Option<CapturePreflightContext>,
}

#[derive(Debug)]
struct CapturePreflightContext {
    expected: ExpectedCaptureSettings,
    state: Arc<(Mutex<CapturePreflightStatus>, Condvar)>,
}

/// Readiness state for expected capture settings checked against one source
/// session. `Compatible` means the source-reported geometry and required
/// controls matched the request before the first read. It does not prove the
/// physical device honored those reports, and table calibration is validated
/// separately by the scene pipeline.
#[derive(Debug, Clone)]
pub enum CapturePreflightStatus {
    Pending {
        session_id: String,
    },
    Compatible(Box<CaptureSessionCompatibility>),
    Rejected {
        session_id: String,
        error: CapturePreflightError,
    },
    Failed {
        session_id: String,
        reason: String,
    },
    Stopped {
        session_id: String,
    },
}

impl CapturePreflightStatus {
    /// Capture session to which this readiness result belongs.
    pub fn session_id(&self) -> &str {
        match self {
            Self::Pending { session_id }
            | Self::Rejected { session_id, .. }
            | Self::Failed { session_id, .. }
            | Self::Stopped { session_id } => session_id,
            Self::Compatible(compatibility) => compatibility.session_id(),
        }
    }

    fn is_ready(&self) -> bool {
        !matches!(self, Self::Pending { .. })
    }
}

/// Bounded outcome of waiting for expected-settings preflight.
#[derive(Debug, Clone)]
pub enum CapturePreflightWait {
    Ready(CapturePreflightStatus),
    TimedOut(CapturePreflightStatus),
}

fn publish_preflight_status(
    state: &Arc<(Mutex<CapturePreflightStatus>, Condvar)>,
    status: CapturePreflightStatus,
) {
    let (lock, changed) = &**state;
    let mut current = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if matches!(*current, CapturePreflightStatus::Pending { .. }) {
        *current = status;
        changed.notify_all();
    }
}

fn preflight_status_snapshot(
    state: &Arc<(Mutex<CapturePreflightStatus>, Condvar)>,
) -> CapturePreflightStatus {
    state
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

fn wait_for_preflight_state(
    state: &Arc<(Mutex<CapturePreflightStatus>, Condvar)>,
    timeout: Duration,
) -> CapturePreflightWait {
    let started_at = Instant::now();
    let (lock, changed) = &**state;
    let mut status = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if status.is_ready() {
            return CapturePreflightWait::Ready(status.clone());
        }
        let remaining = timeout.saturating_sub(started_at.elapsed());
        if remaining.is_zero() {
            return CapturePreflightWait::TimedOut(status.clone());
        }
        let (next_status, result) = changed
            .wait_timeout(status, remaining)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        status = next_status;
        if result.timed_out() && !status.is_ready() {
            return CapturePreflightWait::TimedOut(status.clone());
        }
    }
}

static NEXT_CAPTURE_SESSION_ID: AtomicU64 = AtomicU64::new(0);

fn next_capture_session_id() -> Result<String, CaptureStartError> {
    let previous = NEXT_CAPTURE_SESSION_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| CaptureStartError::SessionIdentityExhausted)?;
    Ok(format!("capture-{}-{}", std::process::id(), previous + 1))
}

/// The single owner thread for one source session and its capacity-one mailbox.
///
/// The factory is unopened until this worker has acquired its shared runtime
/// gate. The returned `CaptureWorker` retains the JoinHandle and gate lease.
/// Callers must keep the same object after a timed-out stop and may not start a
/// replacement with that runtime gate until this worker reports completion.
struct CaptureWorker {
    session_id: String,
    mailbox: LatestFrameMailbox,
    interrupt: Arc<dyn CaptureInterrupt>,
    stop_requested: Arc<AtomicBool>,
    control_failures: Arc<Mutex<Vec<CaptureWorkerFailure>>>,
    completion: Receiver<()>,
    completion_notified: bool,
    join: Option<JoinHandle<CaptureWorkerReport>>,
    owner_lease: Option<CaptureOwnerLease>,
    completed: Option<CaptureWorkerReport>,
    interrupt_succeeded: bool,
    ready_report: Arc<Mutex<Option<CaptureSessionReport>>>,
    preflight_state: Option<Arc<(Mutex<CapturePreflightStatus>, Condvar)>>,
}

impl std::fmt::Debug for CaptureWorker {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CaptureWorker")
            .field("session_id", &self.session_id)
            .field("mailbox", &self.mailbox.stats())
            .field(
                "join_finished",
                &self.join.as_ref().is_none_or(JoinHandle::is_finished),
            )
            .field("completed", &self.completed)
            .finish_non_exhaustive()
    }
}

impl CaptureWorker {
    /// Reserve `owner_gate`, then start one owner thread for an unopened source.
    ///
    /// The gate is acquired before the source factory can open anything. Starts
    /// using the same gate fail before opening if another worker is active or
    /// the gate is quarantined. Each successful start uses its caller-provided
    /// identity or generates a fresh process-local identity.
    #[cfg(test)]
    fn start<F: CaptureSourceFactory>(
        owner_gate: CaptureOwnerGate,
        frame_limit: CaptureFrameLimit,
        frame_budget: CaptureFrameBudget,
        factory: F,
        expected_settings: Option<ExpectedCaptureSettings>,
        requested_session_id: Option<String>,
    ) -> Result<Self, CaptureStartError> {
        Self::start_with_preview(
            owner_gate,
            frame_limit,
            frame_budget,
            factory,
            expected_settings,
            requested_session_id,
            None,
        )
    }

    fn start_with_preview<F: CaptureSourceFactory>(
        owner_gate: CaptureOwnerGate,
        frame_limit: CaptureFrameLimit,
        frame_budget: CaptureFrameBudget,
        factory: F,
        expected_settings: Option<ExpectedCaptureSettings>,
        requested_session_id: Option<String>,
        frame_preview: Option<Arc<LatestCapturedFrameSlot>>,
    ) -> Result<Self, CaptureStartError> {
        let session_id = match requested_session_id {
            Some(session_id) => {
                validate_identifier(&session_id, "session_id")
                    .map_err(CaptureStartError::InvalidSessionIdentity)?;
                session_id
            }
            None => next_capture_session_id()?,
        };
        let mut owner_lease = owner_gate
            .acquire(session_id.clone())
            .map_err(CaptureStartError::OwnerGate)?;
        let interrupt = match catch_unwind(AssertUnwindSafe(|| factory.interrupt_handle())) {
            Ok(interrupt) => interrupt,
            Err(payload) => {
                owner_lease.release_confirmed();
                return Err(CaptureStartError::InterruptHandlePanicked(panic_message(
                    payload,
                )));
            }
        };
        let mailbox = match LatestFrameMailbox::with_budget_and_preview(
            session_id.clone(),
            frame_limit,
            frame_budget.clone(),
            frame_preview,
        ) {
            Ok(mailbox) => mailbox,
            Err(error) => {
                owner_lease.release_confirmed();
                return Err(CaptureStartError::InvalidSessionIdentity(error));
            }
        };
        let stop_requested = Arc::new(AtomicBool::new(false));
        let control_failures = Arc::new(Mutex::new(Vec::new()));
        let ready_report = Arc::new(Mutex::new(None));
        let preflight_state = expected_settings.as_ref().map(|_| {
            Arc::new((
                Mutex::new(CapturePreflightStatus::Pending {
                    session_id: session_id.clone(),
                }),
                Condvar::new(),
            ))
        });
        let input = CaptureWorkerInput {
            factory: Box::new(factory),
            session_id: session_id.clone(),
            frame_limit,
            frame_budget,
            mailbox: mailbox.clone(),
            stop_requested: Arc::clone(&stop_requested),
            ready_report: Arc::clone(&ready_report),
            preflight: expected_settings.map(|expected| CapturePreflightContext {
                expected,
                state: Arc::clone(preflight_state.as_ref().expect("preflight state exists")),
            }),
        };
        let payload = Arc::new(Mutex::new(Some(input)));
        let thread_payload = Arc::clone(&payload);
        let (completion_sender, completion) = mpsc::channel();
        let thread_name = format!("capture-owner-{session_id}");
        let spawn_result = thread::Builder::new().name(thread_name).spawn(move || {
            let input = thread_payload
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            let report = match input {
                Some(input) => run_capture_worker_guarded(input),
                None => CaptureWorkerReport {
                    session_id: "unknown".to_owned(),
                    end: CaptureWorkerEnd::Failed,
                    frames_published: 0,
                    frames_dropped: 0,
                    budget_exhausted_frames: 0,
                    failures: vec![CaptureWorkerFailure::WorkerPanicked(
                        "worker input was missing".to_owned(),
                    )],
                    owner_disposition: CaptureOwnerDisposition::Quarantined {
                        reason: "worker input was missing".to_owned(),
                    },
                },
            };
            let _ = completion_sender.send(());
            report
        });

        let join = match spawn_result {
            Ok(join) => join,
            Err(error) => {
                let input = payload
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                drop(input);
                owner_lease.release_confirmed();
                return Err(CaptureStartError::ThreadSpawn(error));
            }
        };

        Ok(Self {
            session_id,
            mailbox,
            interrupt,
            stop_requested,
            control_failures,
            completion,
            completion_notified: false,
            join: Some(join),
            owner_lease: Some(owner_lease),
            completed: None,
            interrupt_succeeded: false,
            ready_report,
            preflight_state,
        })
    }

    #[cfg(test)]
    fn start_for_test<F: CaptureSourceFactory>(
        owner_gate: CaptureOwnerGate,
        frame_limit: CaptureFrameLimit,
        factory: F,
    ) -> Result<Self, CaptureStartError> {
        Self::start(
            owner_gate,
            frame_limit,
            test_frame_budget(),
            factory,
            None,
            None,
        )
    }

    /// Session identity that the worker supplies to its source on every read.
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn source_report(&self) -> Option<CaptureSessionReport> {
        self.ready_report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Clone the mailbox handle for the inference consumer.
    #[cfg(test)]
    fn mailbox(&self) -> LatestFrameMailbox {
        self.mailbox.clone()
    }

    fn claim_receiver(&self) -> Result<CaptureFrameReceiver, CaptureReceiverClaimError> {
        self.mailbox.claim_receiver()
    }

    /// Request stop, discard pending work, interrupt the source, and wait at
    /// most `timeout` for worker completion after the interrupt call returns.
    ///
    /// On timeout, this method retains the worker thread, source gate lease, and
    /// all worker state. Call `stop` again on the same object or wait for natural
    /// completion; do not construct a replacement source with this runtime gate.
    fn stop(&mut self, timeout: Duration) -> CaptureWorkerWait {
        if !self.is_finished() {
            self.request_stop();
        }
        self.wait_timeout(timeout)
    }

    /// Wait at most `timeout` for natural or requested source completion.
    fn wait_timeout(&mut self, timeout: Duration) -> CaptureWorkerWait {
        if let Some(report) = &self.completed {
            return CaptureWorkerWait::Completed(report.clone());
        }
        let started_at = Instant::now();
        loop {
            if self.is_finished() {
                return self.collect_finished();
            }
            let remaining = timeout.saturating_sub(started_at.elapsed());
            if remaining.is_zero() {
                return CaptureWorkerWait::TimedOut;
            }
            if !self.completion_notified {
                match self.completion.recv_timeout(remaining) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => {
                        self.completion_notified = true;
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if self.is_finished() {
                            return self.collect_finished();
                        }
                        return CaptureWorkerWait::TimedOut;
                    }
                }
            } else {
                thread::yield_now();
            }
        }
    }

    /// Snapshot the worker mailbox's accepted, dropped, and queued frame counts.
    fn mailbox_stats(&self) -> MailboxStats {
        self.mailbox.stats()
    }

    fn is_finished(&self) -> bool {
        self.join.as_ref().is_none_or(JoinHandle::is_finished)
    }

    fn request_stop(&mut self) {
        if self.is_finished() {
            return;
        }
        // Cancellation is the publication barrier: after this lock transition,
        // an in-flight or later read is rejected instead of entering the slot.
        self.mailbox.cancel();
        self.stop_requested.store(true, Ordering::Release);
        if !self.interrupt_succeeded {
            match catch_unwind(AssertUnwindSafe(|| self.interrupt.interrupt())) {
                Ok(()) => self.interrupt_succeeded = true,
                Err(payload) => {
                    self.lock_control_failures()
                        .push(CaptureWorkerFailure::InterruptPanicked(panic_message(
                            payload,
                        )))
                }
            }
        }
    }

    fn collect_finished(&mut self) -> CaptureWorkerWait {
        let Some(join) = self.join.take() else {
            let mut report = self
                .completed
                .clone()
                .unwrap_or_else(|| CaptureWorkerReport {
                    session_id: self.session_id.clone(),
                    end: CaptureWorkerEnd::Failed,
                    frames_published: 0,
                    frames_dropped: 0,
                    budget_exhausted_frames: 0,
                    failures: vec![CaptureWorkerFailure::WorkerPanicked(
                        "worker join handle was missing".to_owned(),
                    )],
                    owner_disposition: CaptureOwnerDisposition::Quarantined {
                        reason: "worker join handle was missing".to_owned(),
                    },
                });
            report
                .failures
                .extend(self.lock_control_failures().drain(..));
            if let Some(mut lease) = self.owner_lease.take() {
                lease.quarantine("worker join handle was missing");
            }
            self.completed = Some(report.clone());
            return CaptureWorkerWait::Completed(report);
        };
        let mut report = match join.join() {
            Ok(report) => report,
            Err(payload) => CaptureWorkerReport {
                session_id: self.session_id.clone(),
                end: CaptureWorkerEnd::Failed,
                frames_published: 0,
                frames_dropped: 0,
                budget_exhausted_frames: 0,
                failures: vec![CaptureWorkerFailure::WorkerPanicked(panic_message(payload))],
                owner_disposition: CaptureOwnerDisposition::Quarantined {
                    reason: "worker panicked before release status was confirmed".to_owned(),
                },
            },
        };
        report
            .failures
            .extend(self.lock_control_failures().drain(..));
        if !report.failures.is_empty() {
            report.end = CaptureWorkerEnd::Failed;
        }
        if let Some(mut lease) = self.owner_lease.take() {
            match &report.owner_disposition {
                CaptureOwnerDisposition::Released => lease.release_confirmed(),
                CaptureOwnerDisposition::Quarantined { reason } => {
                    lease.quarantine(reason.clone());
                }
            }
        }
        self.completed = Some(report.clone());
        CaptureWorkerWait::Completed(report)
    }

    fn lock_control_failures(&self) -> MutexGuard<'_, Vec<CaptureWorkerFailure>> {
        self.control_failures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Drop for CaptureWorker {
    /// Request interruption and join the owner thread before releasing its
    /// gate lease. This intentionally has no timeout: detaching would permit a
    /// second source to open while the old worker might still own a device. A
    /// factory/source that violates the interrupt contract can therefore make
    /// drop block; callers needing bounded shutdown must call `stop(timeout)`
    /// first and retain the worker if it returns `TimedOut`.
    fn drop(&mut self) {
        if !self.is_finished() {
            self.request_stop();
        }
        if self.completed.is_none() {
            if let Some(join) = self.join.take() {
                match join.join() {
                    Ok(mut report) => {
                        report
                            .failures
                            .extend(self.lock_control_failures().drain(..));
                        if !report.failures.is_empty() {
                            report.end = CaptureWorkerEnd::Failed;
                        }
                        if let Some(mut lease) = self.owner_lease.take() {
                            match report.owner_disposition {
                                CaptureOwnerDisposition::Released => lease.release_confirmed(),
                                CaptureOwnerDisposition::Quarantined { reason } => {
                                    lease.quarantine(reason);
                                }
                            }
                        }
                    }
                    Err(payload) => {
                        if let Some(mut lease) = self.owner_lease.take() {
                            lease.quarantine(format!(
                                "worker panicked before release status was confirmed: {}",
                                panic_message(payload)
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// Runtime-owned coordinator for the application's single physical capture source.
///
/// The owner gate and worker are private so callers cannot accidentally create
/// independent gates and pass them to individual starts. Coordinators share a
/// process-wide gate for this single-source product. A timed-out stop retains
/// the worker here, preventing another open through any coordinator until the
/// source thread actually exits.
pub struct CaptureRuntimeCoordinator {
    owner_gate: CaptureOwnerGate,
    frame_limit: CaptureFrameLimit,
    aggregate_frame_limit: CaptureAggregateFrameLimit,
    frame_budget: CaptureFrameBudget,
    worker: Option<CaptureWorker>,
    preflight_state: Option<Arc<(Mutex<CapturePreflightStatus>, Condvar)>>,
    last_report: Option<CaptureWorkerReport>,
}

impl std::fmt::Debug for CaptureRuntimeCoordinator {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CaptureRuntimeCoordinator")
            .field("owner_status", &self.owner_gate.status())
            .field("frame_limit", &self.frame_limit)
            .field("aggregate_frame_limit", &self.aggregate_frame_limit)
            .field("frame_budget", &self.frame_budget.stats())
            .field("preflight_state", &self.preflight_state)
            .field("worker", &self.worker)
            .field("last_report", &self.last_report)
            .finish()
    }
}

/// Result of waiting for the coordinator's current capture worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureRuntimeWait {
    Idle,
    TimedOut,
    Completed(CaptureWorkerReport),
}

impl CaptureRuntimeCoordinator {
    /// Create a coordinator using per-frame and aggregate byte limits resolved
    /// by runtime configuration. Every coordinator in this process shares the
    /// same aggregate budget; a conflicting later configuration is rejected.
    pub fn new(
        frame_limit: CaptureFrameLimit,
        aggregate_limit: CaptureAggregateFrameLimit,
    ) -> Result<Self, CaptureRuntimeConfigurationError> {
        let frame_budget = primary_capture_frame_budget(aggregate_limit)?;
        Ok(Self {
            owner_gate: primary_capture_owner_gate(),
            frame_limit,
            aggregate_frame_limit: aggregate_limit,
            frame_budget,
            worker: None,
            preflight_state: None,
            last_report: None,
        })
    }

    /// Open an unvalidated source through the prototype/replay boundary.
    ///
    /// This low-level path does not compare source metadata with effective
    /// settings before reading. Do not use it for any live provider; use
    /// [`Self::start_preflighted`] when expected capture settings are available.
    pub fn start_unvalidated<F: CaptureSourceFactory>(
        &mut self,
        factory: F,
    ) -> Result<String, CaptureStartError> {
        self.start_inner(factory, None, None, None)
    }

    /// Compatibility alias for prototype/replay callers. This path remains
    /// unvalidated; live acquisition must use [`Self::start_preflighted`].
    pub fn start<F: CaptureSourceFactory>(
        &mut self,
        factory: F,
    ) -> Result<String, CaptureStartError> {
        self.start_unvalidated(factory)
    }

    /// Open a source and validate its opened-session metadata before the first
    /// read. The expected frame limits must exactly match this coordinator's
    /// configured limits. A timeout while awaiting readiness leaves this
    /// coordinator owning the worker and source gate; callers must retain it.
    pub fn start_preflighted<F: CaptureSourceFactory>(
        &mut self,
        expected: ExpectedCaptureSettings,
        factory: F,
    ) -> Result<String, CaptureStartError> {
        self.start_inner(factory, Some(expected), None, None)
    }

    /// Open and preflight one source using a caller-provided session identity.
    ///
    /// This is the process-boundary variant for a supervisor that must bind
    /// the same identity into its protocol hello, source report and every
    /// emitted scene. Invalid identities fail before the source factory opens.
    pub fn start_preflighted_with_session_id<F: CaptureSourceFactory>(
        &mut self,
        expected: ExpectedCaptureSettings,
        session_id: impl Into<String>,
        factory: F,
    ) -> Result<String, CaptureStartError> {
        self.start_inner(factory, Some(expected), Some(session_id.into()), None)
    }

    /// Open a preflighted session and publish each captured frame to an
    /// optional latest-frame preview slot before the inference receiver sees
    /// it. The slot remains absent from capture state when preview is disabled.
    pub fn start_preflighted_with_session_id_and_frame_preview<F: CaptureSourceFactory>(
        &mut self,
        expected: ExpectedCaptureSettings,
        session_id: impl Into<String>,
        frame_preview: Arc<LatestCapturedFrameSlot>,
        factory: F,
    ) -> Result<String, CaptureStartError> {
        self.start_inner(
            factory,
            Some(expected),
            Some(session_id.into()),
            Some(frame_preview),
        )
    }

    fn start_inner<F: CaptureSourceFactory>(
        &mut self,
        factory: F,
        expected_settings: Option<ExpectedCaptureSettings>,
        requested_session_id: Option<String>,
        frame_preview: Option<Arc<LatestCapturedFrameSlot>>,
    ) -> Result<String, CaptureStartError> {
        if let Some(expected) = expected_settings.as_ref() {
            if expected.frame_limit().maximum_rgb8_bytes() != self.frame_limit.maximum_rgb8_bytes()
            {
                return Err(CaptureStartError::PreflightFrameLimitMismatch {
                    expected_bytes: expected.frame_limit().maximum_rgb8_bytes(),
                    coordinator_bytes: self.frame_limit.maximum_rgb8_bytes(),
                });
            }
            if expected.aggregate_frame_limit().maximum_rgb8_bytes()
                != self.aggregate_frame_limit.maximum_rgb8_bytes()
            {
                return Err(CaptureStartError::PreflightAggregateLimitMismatch {
                    expected_bytes: expected.aggregate_frame_limit().maximum_rgb8_bytes(),
                    coordinator_bytes: self.aggregate_frame_limit.maximum_rgb8_bytes(),
                });
            }
        }
        self.reap_completed_worker();
        if let Some(worker) = self.worker.as_ref() {
            return Err(CaptureStartError::WorkerStillRunning {
                session_id: worker.session_id().to_owned(),
            });
        }
        let worker = CaptureWorker::start_with_preview(
            self.owner_gate.clone(),
            self.frame_limit,
            self.frame_budget.clone(),
            factory,
            expected_settings,
            requested_session_id,
            frame_preview,
        )?;
        let session_id = worker.session_id().to_owned();
        self.preflight_state = worker.preflight_state.clone();
        self.worker = Some(worker);
        self.last_report = None;
        Ok(session_id)
    }

    /// Claim the sole consumer handle for the active source session.
    ///
    /// The receiver is not cloneable, and each frame is exposed only as a
    /// synchronous callback borrow. A second claim for the same session fails
    /// even if the first receiver has since been dropped.
    pub fn claim_receiver(&self) -> Result<CaptureFrameReceiver, CaptureReceiverClaimError> {
        self.worker
            .as_ref()
            .ok_or(CaptureReceiverClaimError::NoActiveSession)?
            .claim_receiver()
    }

    /// Return metadata for the active source, once open has succeeded.
    ///
    /// The snapshot is scoped to the active worker's generated session ID. It
    /// is cleared from the active view as soon as the worker is reaped, and a
    /// failed or interrupted open never makes a report available.
    pub fn active_source_report(&mut self) -> Option<CaptureSessionReport> {
        self.reap_completed_worker();
        self.worker.as_ref().and_then(CaptureWorker::source_report)
    }

    /// Snapshot expected-settings readiness for the most recently started
    /// preflighted session. Compatibility reflects only reported metadata.
    pub fn preflight_status(&self) -> Option<CapturePreflightStatus> {
        self.preflight_state.as_ref().map(preflight_status_snapshot)
    }

    /// Wait at most `timeout` for the current preflighted session to become
    /// compatible, rejected, failed, or stopped. A timeout is only a
    /// observation: it never releases or transfers worker ownership.
    pub fn wait_for_preflight(&self, timeout: Duration) -> Option<CapturePreflightWait> {
        self.preflight_state
            .as_ref()
            .map(|state| wait_for_preflight_state(state, timeout))
    }

    /// Snapshot queue depth and frame acceptance/loss counts for the active source.
    pub fn mailbox_stats(&self) -> Option<MailboxStats> {
        self.worker.as_ref().map(CaptureWorker::mailbox_stats)
    }

    /// Snapshot the process-wide frame budget shared by all coordinators.
    pub fn frame_budget_stats(&self) -> CaptureFrameBudgetStats {
        self.frame_budget.stats()
    }

    /// Request stop and wait no longer than `timeout` for the capture worker
    /// after the adapter's synchronous interrupt call returns. The adapter must
    /// keep that call prompt. A timeout leaves the worker and owner lease
    /// retained in this coordinator; dropping it later joins without a timeout.
    pub fn stop(&mut self, timeout: Duration) -> CaptureRuntimeWait {
        let Some(worker) = self.worker.as_mut() else {
            return CaptureRuntimeWait::Idle;
        };
        match worker.stop(timeout) {
            CaptureWorkerWait::TimedOut => CaptureRuntimeWait::TimedOut,
            CaptureWorkerWait::Completed(report) => {
                self.last_report = Some(report.clone());
                self.worker = None;
                CaptureRuntimeWait::Completed(report)
            }
        }
    }

    /// Wait for natural or requested completion without requesting a stop.
    pub fn wait_timeout(&mut self, timeout: Duration) -> CaptureRuntimeWait {
        let Some(worker) = self.worker.as_mut() else {
            return CaptureRuntimeWait::Idle;
        };
        match worker.wait_timeout(timeout) {
            CaptureWorkerWait::TimedOut => CaptureRuntimeWait::TimedOut,
            CaptureWorkerWait::Completed(report) => {
                self.last_report = Some(report.clone());
                self.worker = None;
                CaptureRuntimeWait::Completed(report)
            }
        }
    }

    /// Report exclusive-source status without changing ownership.
    pub fn owner_status(&self) -> CaptureOwnerStatus {
        self.owner_gate.status()
    }

    /// Most recently completed worker report, if one has completed.
    pub fn last_report(&self) -> Option<&CaptureWorkerReport> {
        self.last_report.as_ref()
    }

    fn reap_completed_worker(&mut self) {
        let completed =
            self.worker
                .as_mut()
                .and_then(|worker| match worker.wait_timeout(Duration::ZERO) {
                    CaptureWorkerWait::Completed(report) => Some(report),
                    CaptureWorkerWait::TimedOut => None,
                });
        if let Some(report) = completed {
            self.last_report = Some(report);
            self.worker = None;
        }
    }
}

fn run_capture_worker(input: CaptureWorkerInput) -> CaptureWorkerReport {
    let CaptureWorkerInput {
        factory,
        session_id,
        frame_limit,
        frame_budget,
        mailbox,
        stop_requested,
        ready_report,
        preflight,
    } = input;
    if stop_requested.load(Ordering::Acquire) {
        // A stop can arrive after the thread is spawned but before it starts.
        // Do not open hardware after the caller has already cancelled startup.
        mailbox.cancel();
        if let Some(context) = &preflight {
            publish_preflight_status(
                &context.state,
                CapturePreflightStatus::Stopped {
                    session_id: session_id.clone(),
                },
            );
        }
        return CaptureWorkerReport {
            session_id,
            end: CaptureWorkerEnd::Stopped,
            frames_published: 0,
            frames_dropped: 0,
            budget_exhausted_frames: 0,
            failures: Vec::new(),
            owner_disposition: CaptureOwnerDisposition::Released,
        };
    }
    let open_result = catch_unwind(AssertUnwindSafe(|| {
        factory.open_with_session_id(&session_id)
    }));
    match open_result {
        Err(payload) => {
            let failure = CaptureWorkerFailure::OpenPanicked(panic_message(payload));
            let reason = failure.to_string();
            mailbox.fail(reason.clone());
            if let Some(context) = &preflight {
                publish_preflight_status(
                    &context.state,
                    CapturePreflightStatus::Failed {
                        session_id: session_id.clone(),
                        reason: reason.clone(),
                    },
                );
            }
            CaptureWorkerReport {
                session_id,
                end: CaptureWorkerEnd::Failed,
                frames_published: 0,
                frames_dropped: 0,
                budget_exhausted_frames: 0,
                failures: vec![failure],
                owner_disposition: CaptureOwnerDisposition::Quarantined { reason },
            }
        }
        Ok(Err(CaptureOpenError::Failed(message))) => {
            // The factory contract requires a failed open to clean up any
            // partial device acquisition before returning an error.
            let failure = CaptureWorkerFailure::Open(message);
            mailbox.fail(failure.to_string());
            if let Some(context) = &preflight {
                publish_preflight_status(
                    &context.state,
                    CapturePreflightStatus::Failed {
                        session_id: session_id.clone(),
                        reason: failure.to_string(),
                    },
                );
            }
            CaptureWorkerReport {
                session_id,
                end: CaptureWorkerEnd::Failed,
                frames_published: 0,
                frames_dropped: 0,
                budget_exhausted_frames: 0,
                failures: vec![failure],
                owner_disposition: CaptureOwnerDisposition::Released,
            }
        }
        Ok(Err(CaptureOpenError::Interrupted)) => {
            if stop_requested.load(Ordering::Acquire) {
                mailbox.cancel();
                if let Some(context) = &preflight {
                    publish_preflight_status(
                        &context.state,
                        CapturePreflightStatus::Stopped {
                            session_id: session_id.clone(),
                        },
                    );
                }
                CaptureWorkerReport {
                    session_id,
                    end: CaptureWorkerEnd::Stopped,
                    frames_published: 0,
                    frames_dropped: 0,
                    budget_exhausted_frames: 0,
                    failures: Vec::new(),
                    owner_disposition: CaptureOwnerDisposition::Released,
                }
            } else {
                let failure = CaptureWorkerFailure::OpenInterrupted;
                mailbox.fail(failure.to_string());
                if let Some(context) = &preflight {
                    publish_preflight_status(
                        &context.state,
                        CapturePreflightStatus::Failed {
                            session_id: session_id.clone(),
                            reason: failure.to_string(),
                        },
                    );
                }
                CaptureWorkerReport {
                    session_id,
                    end: CaptureWorkerEnd::Failed,
                    frames_published: 0,
                    frames_dropped: 0,
                    budget_exhausted_frames: 0,
                    failures: vec![failure],
                    owner_disposition: CaptureOwnerDisposition::Released,
                }
            }
        }
        Ok(Ok(mut source)) => {
            let loop_result = if stop_requested.load(Ordering::Acquire) {
                // Stop may race a successful open. Do not query metadata or
                // read a frame after cancellation; release still runs below.
                if let Some(context) = &preflight {
                    publish_preflight_status(
                        &context.state,
                        CapturePreflightStatus::Stopped {
                            session_id: session_id.clone(),
                        },
                    );
                }
                Ok(CaptureReadLoopResult {
                    end: CaptureWorkerEnd::Stopped,
                    failures: Vec::new(),
                    frames_published: 0,
                    frames_dropped: 0,
                    budget_exhausted_frames: 0,
                })
            } else {
                match catch_unwind(AssertUnwindSafe(|| source.metadata())) {
                    Ok(metadata) => {
                        let timestamp_semantics = metadata.timestamp_semantics().clone();
                        if stop_requested.load(Ordering::Acquire) {
                            if let Some(context) = &preflight {
                                publish_preflight_status(
                                    &context.state,
                                    CapturePreflightStatus::Stopped {
                                        session_id: session_id.clone(),
                                    },
                                );
                            }
                            Ok(CaptureReadLoopResult {
                                end: CaptureWorkerEnd::Stopped,
                                failures: Vec::new(),
                                frames_published: 0,
                                frames_dropped: 0,
                                budget_exhausted_frames: 0,
                            })
                        } else {
                            let report = CaptureSessionReport::new(session_id.clone(), metadata);
                            *ready_report
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                Some(report.clone());
                            let preflight_error = preflight.as_ref().and_then(|context| {
                                match context.expected.validate_session(&report) {
                                    Ok(compatibility) => {
                                        match mailbox.bind_frame_preview(&compatibility) {
                                            Ok(()) => {
                                                publish_preflight_status(
                                                    &context.state,
                                                    CapturePreflightStatus::Compatible(Box::new(
                                                        compatibility,
                                                    )),
                                                );
                                                None
                                            }
                                            Err(error) => {
                                                let reason = format!(
                                                    "frame preview could not bind to accepted capture session: {error}"
                                                );
                                                publish_preflight_status(
                                                    &context.state,
                                                    CapturePreflightStatus::Failed {
                                                        session_id: session_id.clone(),
                                                        reason: reason.clone(),
                                                    },
                                                );
                                                mailbox.fail(reason.clone());
                                                Some(CaptureWorkerFailure::FramePreviewSetup(reason))
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        publish_preflight_status(
                                            &context.state,
                                            CapturePreflightStatus::Rejected {
                                                session_id: session_id.clone(),
                                                error: error.clone(),
                                            },
                                        );
                                        // Publish failure as soon as the
                                        // mismatch is known. Source release
                                        // still runs on this owner thread, but
                                        // consumers must not wait on a slow
                                        // release operation to learn no frame
                                        // can arrive from this session.
                                        mailbox.fail(error.to_string());
                                        Some(CaptureWorkerFailure::PreflightRejected(error))
                                    }
                                }
                            });
                            if let Some(failure) = preflight_error {
                                Ok(CaptureReadLoopResult {
                                    end: CaptureWorkerEnd::Failed,
                                    failures: vec![failure],
                                    frames_published: 0,
                                    frames_dropped: 0,
                                    budget_exhausted_frames: 0,
                                })
                            } else {
                                let expected_output_geometry = preflight.as_ref().map(|context| {
                                    context.expected.output_frame_geometry().unwrap_or_else(|| {
                                        let profile =
                                            context.expected.expected_negotiated_profile();
                                        CaptureOutputFrameGeometry::new(
                                            profile.width_px(),
                                            profile.height_px(),
                                        )
                                        .expect("validated profile dimensions are non-zero")
                                    })
                                });
                                catch_unwind(AssertUnwindSafe(|| {
                                    run_capture_read_loop(
                                        &mut *source,
                                        CaptureReadContext {
                                            session_id: &session_id,
                                            frame_limit,
                                            frame_budget: &frame_budget,
                                            timestamp_semantics: &timestamp_semantics,
                                            expected_output_geometry,
                                            mailbox: &mailbox,
                                            stop_requested: &stop_requested,
                                        },
                                    )
                                }))
                            }
                        }
                    }
                    Err(payload) => {
                        let failure =
                            CaptureWorkerFailure::SourceMetadataPanicked(panic_message(payload));
                        if let Some(context) = &preflight {
                            publish_preflight_status(
                                &context.state,
                                CapturePreflightStatus::Failed {
                                    session_id: session_id.clone(),
                                    reason: failure.to_string(),
                                },
                            );
                        }
                        Ok(CaptureReadLoopResult {
                            end: CaptureWorkerEnd::Failed,
                            failures: vec![failure],
                            frames_published: 0,
                            frames_dropped: 0,
                            budget_exhausted_frames: 0,
                        })
                    }
                }
            };
            let (
                mut end,
                mut failures,
                frames_published,
                frames_dropped,
                budget_exhausted_frames,
                worker_panicked,
            ) = match loop_result {
                Ok(result) => (
                    result.end,
                    result.failures,
                    result.frames_published,
                    result.frames_dropped,
                    result.budget_exhausted_frames,
                    false,
                ),
                Err(payload) => (
                    CaptureWorkerEnd::Failed,
                    vec![CaptureWorkerFailure::WorkerPanicked(panic_message(payload))],
                    0,
                    0,
                    0,
                    true,
                ),
            };
            let release_result = catch_unwind(AssertUnwindSafe(|| source.release()));
            let release_confirmed = match release_result {
                Ok(Ok(())) => true,
                Ok(Err(message)) => {
                    failures.push(CaptureWorkerFailure::Release(message));
                    false
                }
                Err(payload) => {
                    failures.push(CaptureWorkerFailure::ReleasePanicked(panic_message(
                        payload,
                    )));
                    false
                }
            };
            let source_drop_panicked = catch_unwind(AssertUnwindSafe(|| drop(source))).is_err();
            if source_drop_panicked {
                failures.push(CaptureWorkerFailure::SourceDropPanicked(
                    "source destructor panicked".to_owned(),
                ));
            }
            let disposition = if worker_panicked {
                CaptureOwnerDisposition::Quarantined {
                    reason: "worker panicked before release was confirmed".to_owned(),
                }
            } else if !release_confirmed {
                CaptureOwnerDisposition::Quarantined {
                    reason: "source release was not safely confirmed".to_owned(),
                }
            } else if source_drop_panicked {
                CaptureOwnerDisposition::Quarantined {
                    reason: "source destructor panicked after release".to_owned(),
                }
            } else {
                CaptureOwnerDisposition::Released
            };

            if !failures.is_empty() {
                end = CaptureWorkerEnd::Failed;
            } else if end == CaptureWorkerEnd::NaturalEnd && stop_requested.load(Ordering::Acquire)
            {
                // A stop requested while release was running supersedes a
                // natural-end status and leaves the mailbox cancelled.
                end = CaptureWorkerEnd::Stopped;
            }
            match end {
                CaptureWorkerEnd::NaturalEnd => {
                    mailbox.close();
                }
                CaptureWorkerEnd::Stopped => {
                    mailbox.cancel();
                }
                CaptureWorkerEnd::Failed => {
                    let reason = failures
                        .first()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "capture worker failed".to_owned());
                    mailbox.fail(reason);
                }
            }
            CaptureWorkerReport {
                session_id,
                end,
                frames_published,
                frames_dropped,
                budget_exhausted_frames,
                failures,
                owner_disposition: disposition,
            }
        }
    }
}

/// Convert an unexpected panic escaping the source lifecycle into a report,
/// wake consumers, and leave the owner gate quarantined when the caller joins.
fn run_capture_worker_guarded(input: CaptureWorkerInput) -> CaptureWorkerReport {
    let session_id = input.session_id.clone();
    let mailbox = input.mailbox.clone();
    let preflight_state = input
        .preflight
        .as_ref()
        .map(|context| Arc::clone(&context.state));
    match catch_unwind(AssertUnwindSafe(|| run_capture_worker(input))) {
        Ok(report) => report,
        Err(payload) => {
            let failure = CaptureWorkerFailure::WorkerPanicked(panic_message(payload));
            let reason = failure.to_string();
            mailbox.fail(reason);
            if let Some(state) = preflight_state {
                publish_preflight_status(
                    &state,
                    CapturePreflightStatus::Failed {
                        session_id: session_id.clone(),
                        reason: failure.to_string(),
                    },
                );
            }
            CaptureWorkerReport {
                session_id,
                end: CaptureWorkerEnd::Failed,
                frames_published: 0,
                frames_dropped: 0,
                budget_exhausted_frames: 0,
                failures: vec![failure],
                owner_disposition: CaptureOwnerDisposition::Quarantined {
                    reason: "worker panicked before release status was confirmed".to_owned(),
                },
            }
        }
    }
}

struct CaptureReadLoopResult {
    end: CaptureWorkerEnd,
    failures: Vec<CaptureWorkerFailure>,
    frames_published: u64,
    frames_dropped: u64,
    budget_exhausted_frames: u64,
}

#[derive(Clone, Copy)]
struct CaptureReadContext<'a> {
    session_id: &'a str,
    frame_limit: CaptureFrameLimit,
    frame_budget: &'a CaptureFrameBudget,
    timestamp_semantics: &'a CaptureTimestampSemantics,
    expected_output_geometry: Option<CaptureOutputFrameGeometry>,
    mailbox: &'a LatestFrameMailbox,
    stop_requested: &'a AtomicBool,
}

fn run_capture_read_loop(
    source: &mut dyn CaptureSource,
    context: CaptureReadContext<'_>,
) -> CaptureReadLoopResult {
    let CaptureReadContext {
        session_id,
        frame_limit,
        frame_budget,
        timestamp_semantics,
        expected_output_geometry,
        mailbox,
        stop_requested,
    } = context;
    let mut frames_published = 0_u64;
    let mut frames_dropped = 0_u64;
    let mut budget_exhausted_frames = 0_u64;
    loop {
        if stop_requested.load(Ordering::Acquire) {
            return CaptureReadLoopResult {
                end: CaptureWorkerEnd::Stopped,
                failures: Vec::new(),
                frames_published,
                frames_dropped,
                budget_exhausted_frames,
            };
        }
        let read_result = catch_unwind(AssertUnwindSafe(|| {
            source.read(session_id, frame_limit, frame_budget)
        }));
        match read_result {
            Err(payload) => {
                return CaptureReadLoopResult {
                    end: CaptureWorkerEnd::Failed,
                    failures: vec![CaptureWorkerFailure::ReadPanicked(panic_message(payload))],
                    frames_published,
                    frames_dropped,
                    budget_exhausted_frames,
                };
            }
            Ok(Err(message)) => {
                return CaptureReadLoopResult {
                    end: CaptureWorkerEnd::Failed,
                    failures: vec![CaptureWorkerFailure::Read(message)],
                    frames_published,
                    frames_dropped,
                    budget_exhausted_frames,
                };
            }
            Ok(Ok(CaptureReadOutcome::End)) => {
                return CaptureReadLoopResult {
                    end: if stop_requested.load(Ordering::Acquire) {
                        CaptureWorkerEnd::Stopped
                    } else {
                        CaptureWorkerEnd::NaturalEnd
                    },
                    failures: Vec::new(),
                    frames_published,
                    frames_dropped,
                    budget_exhausted_frames,
                };
            }
            Ok(Ok(CaptureReadOutcome::Dropped(reason))) => {
                frames_dropped = frames_dropped.saturating_add(1);
                if matches!(reason, CaptureFrameDropReason::BudgetExhausted(_)) {
                    budget_exhausted_frames = budget_exhausted_frames.saturating_add(1);
                }
            }
            Ok(Ok(CaptureReadOutcome::Frame(frame))) => {
                if let Some(geometry) = expected_output_geometry {
                    let source = frame.source();
                    if source.width_px != geometry.width_px()
                        || source.height_px != geometry.height_px()
                    {
                        return CaptureReadLoopResult {
                            end: CaptureWorkerEnd::Failed,
                            failures: vec![CaptureWorkerFailure::FrameProfileMismatch {
                                expected_width_px: geometry.width_px(),
                                expected_height_px: geometry.height_px(),
                                actual_width_px: source.width_px,
                                actual_height_px: source.height_px,
                            }],
                            frames_published,
                            frames_dropped,
                            budget_exhausted_frames,
                        };
                    }
                }
                let actual_timestamp = frame.source().timestamp.as_ref();
                if !frame_timestamp_matches_semantics(timestamp_semantics, actual_timestamp) {
                    return CaptureReadLoopResult {
                        end: CaptureWorkerEnd::Failed,
                        failures: vec![CaptureWorkerFailure::FrameTimestampMismatch {
                            expected: timestamp_semantics.clone(),
                            actual: actual_timestamp.map(|timestamp| {
                                CaptureTimestampSemantics::SourceTimestamp {
                                    unit: timestamp.unit,
                                    clock: timestamp.clock,
                                    domain_id: timestamp.domain_id.clone(),
                                    stage: timestamp.stage,
                                }
                            }),
                        }],
                        frames_published,
                        frames_dropped,
                        budget_exhausted_frames,
                    };
                }

                match mailbox.publish(frame) {
                    Ok(()) => frames_published = frames_published.saturating_add(1),
                    Err(PublishError::Terminal(MailboxTerminal::Cancelled)) => {
                        return CaptureReadLoopResult {
                            end: CaptureWorkerEnd::Stopped,
                            failures: Vec::new(),
                            frames_published,
                            frames_dropped,
                            budget_exhausted_frames,
                        };
                    }
                    Err(error) => {
                        return CaptureReadLoopResult {
                            end: CaptureWorkerEnd::Failed,
                            failures: vec![CaptureWorkerFailure::PublishRejected(error)],
                            frames_published,
                            frames_dropped,
                            budget_exhausted_frames,
                        };
                    }
                }
            }
        }
    }
}

fn frame_timestamp_matches_semantics(
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

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        test_frame_budget, BudgetedRgb8Buffer, CaptureAggregateFrameLimit, CaptureEnvelopeError,
        CaptureFrameBudget, CaptureFrameLimit, LatestFrameMailbox, MailboxTerminal, ReceiveOutcome,
        RgbFrameEnvelope, TEST_FRAME_LIMIT,
    };
    use crate::detection_batch::{
        BatchTimestamp, ClockKind, SourceFrame, TimeUnit, TimestampStage,
    };
    use std::thread;
    use std::time::{Duration, Instant};

    fn timestamp(stage: TimestampStage) -> BatchTimestamp {
        BatchTimestamp {
            ticks: "123456".to_owned(),
            unit: TimeUnit::Nanoseconds,
            clock: ClockKind::Sensor,
            domain_id: "camera-clock".to_owned(),
            stage,
        }
    }

    fn envelope(session_id: &str, frame_id: &str, fill: u8) -> RgbFrameEnvelope {
        envelope_from_budget(
            &test_frame_budget(),
            session_id,
            frame_id,
            fill,
            2,
            1,
            TEST_FRAME_LIMIT,
        )
        .expect("valid frame envelope")
    }

    pub(super) fn envelope_from_budget(
        budget: &CaptureFrameBudget,
        session_id: &str,
        frame_id: &str,
        fill: u8,
        width_px: u32,
        height_px: u32,
        frame_limit: CaptureFrameLimit,
    ) -> Result<RgbFrameEnvelope, CaptureEnvelopeError> {
        let mut pixels = budget
            .allocate_rgb8_buffer(width_px, height_px, frame_limit)
            .map_err(|error| match error {
                super::CaptureFrameAllocationError::InvalidFrame(error) => error,
                other => panic!("test frame allocation failed: {other}"),
            })?;
        pixels.as_mut_slice().fill(fill);
        RgbFrameEnvelope::new(
            SourceFrame {
                session_id: session_id.to_owned(),
                frame_id: frame_id.to_owned(),
                timestamp: Some(timestamp(TimestampStage::Exposure)),
                width_px,
                height_px,
            },
            pixels,
            Instant::now(),
            frame_limit,
        )
    }

    fn test_pixels(width_px: u32, height_px: u32, limit: CaptureFrameLimit) -> BudgetedRgb8Buffer {
        test_frame_budget()
            .allocate_rgb8_buffer(width_px, height_px, limit)
            .expect("test pixel buffer is valid")
    }

    fn wait_for_receivers(mailbox: &LatestFrameMailbox, expected: usize, timeout: Duration) {
        let started_at = Instant::now();
        loop {
            if mailbox.stats().waiting_receivers >= expected {
                return;
            }
            assert!(
                started_at.elapsed() < timeout,
                "timed out waiting for {expected} blocked receiver(s)"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn envelope_preserves_source_clock_and_separate_host_read_time() {
        let source_time = timestamp(TimestampStage::SharedMemoryPublication);
        let host_read_completed_at = Instant::now();
        let frame = RgbFrameEnvelope::new(
            SourceFrame {
                session_id: "session-a".to_owned(),
                frame_id: "frame-1".to_owned(),
                timestamp: Some(source_time.clone()),
                width_px: 1,
                height_px: 1,
            },
            {
                let mut pixels = test_pixels(1, 1, TEST_FRAME_LIMIT);
                pixels.as_mut_slice().copy_from_slice(&[1, 2, 3]);
                pixels
            },
            host_read_completed_at,
            TEST_FRAME_LIMIT,
        )
        .unwrap();

        assert_eq!(frame.source.timestamp, Some(source_time));
        assert_eq!(frame.host_read_completed_at, host_read_completed_at);
        assert_eq!(frame.rgb8(), &[1, 2, 3]);
    }

    #[test]
    fn aggregate_budget_reserves_before_allocation_and_releases_after_exhaustion() {
        let budget = CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(6).unwrap());
        let buffer = budget
            .allocate_rgb8_buffer(2, 1, TEST_FRAME_LIMIT)
            .expect("exact configured aggregate limit is allocatable");
        assert_eq!(buffer.capacity(), 6);
        assert_eq!(budget.stats().current_bytes, buffer.capacity());
        assert_eq!(budget.stats().high_water_mark_bytes, buffer.capacity());

        assert_eq!(
            budget
                .allocate_rgb8_buffer(1, 1, TEST_FRAME_LIMIT)
                .unwrap_err(),
            super::CaptureFrameAllocationError::BudgetExhausted(super::CaptureBudgetExhaustion {
                requested_bytes: 3,
                current_bytes: 6,
                available_bytes: 0,
                configured_limit_bytes: 6,
            })
        );
        assert_eq!(budget.stats().rejected_allocations, 1);

        drop(buffer);
        assert_eq!(budget.stats().current_bytes, 0);
        assert_eq!(
            budget
                .allocate_rgb8_buffer(2, 1, TEST_FRAME_LIMIT)
                .unwrap()
                .len(),
            6
        );
    }

    #[test]
    fn capacity_growth_rejection_releases_the_original_reservation() {
        let budget = CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(6).unwrap());
        let mut reservation = budget.reserve(6).unwrap();
        assert!(reservation.set_capacity(7).is_err());
        assert_eq!(budget.stats().current_bytes, 6);
        assert_eq!(budget.stats().rejected_allocations, 1);

        drop(reservation);
        assert_eq!(budget.stats().current_bytes, 0);
    }

    #[test]
    fn mailbox_rejects_envelopes_budgeted_outside_its_runtime() {
        let runtime_budget = test_frame_budget();
        let mailbox =
            LatestFrameMailbox::with_budget("session-a", TEST_FRAME_LIMIT, runtime_budget.clone())
                .unwrap();
        let foreign_budget = CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(6).unwrap());
        let foreign = envelope_from_budget(
            &foreign_budget,
            "session-a",
            "foreign-budget",
            1,
            2,
            1,
            TEST_FRAME_LIMIT,
        )
        .unwrap();

        assert_eq!(
            mailbox.publish(foreign),
            Err(super::PublishError::ForeignBudget)
        );
        assert_eq!(mailbox.stats().rejected_frames, 1);
        assert_eq!(runtime_budget.stats().current_bytes, 0);
        assert_eq!(foreign_budget.stats().current_bytes, 0);
    }

    #[test]
    fn actual_capacity_is_charged_and_envelope_replacement_and_cancellation_release_it() {
        let budget = CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(12).unwrap());
        let first = envelope_from_budget(
            &budget,
            "session-budget",
            "frame-1",
            1,
            2,
            1,
            TEST_FRAME_LIMIT,
        )
        .unwrap();
        assert_eq!(budget.stats().current_bytes, first.rgb8.capacity());
        drop(first);
        assert_eq!(budget.stats().current_bytes, 0);

        let mailbox =
            LatestFrameMailbox::with_budget("session-budget", TEST_FRAME_LIMIT, budget.clone())
                .unwrap();
        let first = envelope_from_budget(
            &budget,
            "session-budget",
            "frame-2",
            2,
            2,
            1,
            TEST_FRAME_LIMIT,
        )
        .unwrap();
        let first_capacity = first.rgb8.capacity();
        mailbox.publish(first).unwrap();
        assert_eq!(budget.stats().current_bytes, first_capacity);

        let second = envelope_from_budget(
            &budget,
            "session-budget",
            "frame-3",
            3,
            2,
            1,
            TEST_FRAME_LIMIT,
        )
        .unwrap();
        let second_capacity = second.rgb8.capacity();
        assert_eq!(
            budget.stats().current_bytes,
            first_capacity + second_capacity
        );
        mailbox.publish(second).unwrap();
        assert_eq!(budget.stats().current_bytes, second_capacity);
        assert_eq!(mailbox.stats().replaced_frames, 1);

        mailbox.cancel();
        assert_eq!(mailbox.stats().cancelled_frames, 1);
        assert_eq!(budget.stats().current_bytes, 0);
    }

    #[test]
    fn configured_frame_limit_is_positive_and_checked_before_allocation() {
        assert_eq!(
            CaptureFrameLimit::new(0),
            Err(super::CaptureFrameLimitError::Zero)
        );
        let exact_limit = CaptureFrameLimit::new(6).unwrap();
        assert_eq!(
            RgbFrameEnvelope::required_rgb8_len(2, 1, exact_limit),
            Ok(6)
        );
        let budget = CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(6).unwrap());
        let allocated = budget.allocate_rgb8_buffer(2, 1, exact_limit).unwrap();
        assert_eq!(allocated.len(), 6);
        assert!(allocated.capacity() <= exact_limit.maximum_rgb8_bytes());
        assert_eq!(budget.stats().current_bytes, allocated.capacity());
        assert!(matches!(
            budget.allocate_rgb8_buffer(3, 1, exact_limit),
            Err(super::CaptureFrameAllocationError::InvalidFrame(
                CaptureEnvelopeError::FrameExceedsLimit {
                    required: 9,
                    maximum: 6
                }
            ))
        ));

        let smaller_limit = CaptureFrameLimit::new(5).unwrap();
        assert_eq!(
            RgbFrameEnvelope::required_rgb8_len(2, 1, smaller_limit),
            Err(CaptureEnvelopeError::FrameExceedsLimit {
                required: 6,
                maximum: 5
            })
        );
        assert!(matches!(
            RgbFrameEnvelope::required_rgb8_len(u32::MAX, u32::MAX, exact_limit),
            Err(CaptureEnvelopeError::ImageSizeOverflow { .. })
        ));
    }

    #[test]
    fn envelope_and_mailbox_both_enforce_the_configured_frame_limit() {
        let oversized_source = envelope("session-a", "frame-large", 0).source;
        let small_limit = CaptureFrameLimit::new(5).unwrap();
        let frame = RgbFrameEnvelope::new(
            oversized_source,
            test_pixels(2, 1, CaptureFrameLimit::new(64).unwrap()),
            Instant::now(),
            CaptureFrameLimit::new(64).unwrap(),
        )
        .unwrap();
        let mailbox = LatestFrameMailbox::new("session-a", small_limit).unwrap();
        assert_eq!(
            mailbox.publish(frame),
            Err(super::PublishError::FrameExceedsLimit {
                actual: 6,
                maximum: 5
            })
        );
        assert_eq!(mailbox.stats().rejected_frames, 1);
    }

    #[test]
    fn envelope_accepts_missing_source_timestamp_without_synthesizing_one() {
        let mut source = envelope("session-a", "frame-1", 0).source;
        source.timestamp = None;
        let frame = RgbFrameEnvelope::new(
            source,
            test_pixels(2, 1, TEST_FRAME_LIMIT),
            Instant::now(),
            TEST_FRAME_LIMIT,
        )
        .unwrap();
        assert_eq!(frame.source.timestamp, None);
    }

    #[test]
    fn envelope_rejects_invalid_identifiers_dimensions_and_pixel_lengths() {
        let mut source = envelope("session-a", "frame-1", 0).source;
        source.session_id.clear();
        assert_eq!(
            RgbFrameEnvelope::new(
                source,
                test_pixels(2, 1, TEST_FRAME_LIMIT),
                Instant::now(),
                TEST_FRAME_LIMIT,
            ),
            Err(CaptureEnvelopeError::InvalidIdentifier {
                field: "source.session_id"
            })
        );

        let mut source = envelope("session-a", "frame-1", 0).source;
        source.width_px = 0;
        assert!(matches!(
            RgbFrameEnvelope::new(
                source,
                test_pixels(1, 1, TEST_FRAME_LIMIT),
                Instant::now(),
                TEST_FRAME_LIMIT,
            ),
            Err(CaptureEnvelopeError::InvalidDimensions { .. })
        ));

        let source = envelope("session-a", "frame-1", 0).source;
        assert_eq!(
            RgbFrameEnvelope::new(
                source,
                test_pixels(1, 1, TEST_FRAME_LIMIT),
                Instant::now(),
                TEST_FRAME_LIMIT,
            ),
            Err(CaptureEnvelopeError::PixelLengthMismatch {
                expected: 6,
                actual: 3
            })
        );

        let mut source = envelope("session-a", "frame-1", 0).source;
        source.width_px = u32::MAX;
        source.height_px = u32::MAX;
        assert!(matches!(
            RgbFrameEnvelope::new(
                source,
                test_pixels(1, 1, TEST_FRAME_LIMIT),
                Instant::now(),
                TEST_FRAME_LIMIT,
            ),
            Err(CaptureEnvelopeError::ImageSizeOverflow { .. })
        ));
    }

    #[test]
    fn envelope_rejects_inference_and_batch_publication_as_source_time() {
        for stage in [
            TimestampStage::InferenceStarted,
            TimestampStage::InferenceComplete,
            TimestampStage::BatchPublication,
        ] {
            let mut source = envelope("session-a", "frame-1", 0).source;
            source.timestamp = Some(timestamp(stage));
            assert_eq!(
                RgbFrameEnvelope::new(
                    source,
                    test_pixels(2, 1, TEST_FRAME_LIMIT),
                    Instant::now(),
                    TEST_FRAME_LIMIT,
                ),
                Err(CaptureEnvelopeError::InvalidSourceTimestampStage(stage))
            );
        }
    }

    #[test]
    fn envelope_rejects_malformed_timestamp_values() {
        let mut source = envelope("session-a", "frame-1", 0).source;
        let mut bad = timestamp(TimestampStage::Exposure);
        bad.ticks = "1.5".to_owned();
        source.timestamp = Some(bad);
        assert_eq!(
            RgbFrameEnvelope::new(
                source,
                test_pixels(2, 1, TEST_FRAME_LIMIT),
                Instant::now(),
                TEST_FRAME_LIMIT,
            ),
            Err(CaptureEnvelopeError::InvalidSourceTimestampTicks)
        );

        let mut source = envelope("session-a", "frame-1", 0).source;
        let mut bad = timestamp(TimestampStage::Exposure);
        bad.domain_id = "\n".to_owned();
        source.timestamp = Some(bad);
        assert_eq!(
            RgbFrameEnvelope::new(
                source,
                test_pixels(2, 1, TEST_FRAME_LIMIT),
                Instant::now(),
                TEST_FRAME_LIMIT,
            ),
            Err(CaptureEnvelopeError::InvalidSourceTimestampDomain)
        );
    }

    #[test]
    fn mailbox_replaces_old_frames_and_never_exceeds_capacity_one() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        mailbox
            .publish(envelope("session-a", "frame-1", 1))
            .unwrap();
        mailbox
            .publish(envelope("session-a", "frame-2", 2))
            .unwrap();
        mailbox
            .publish(envelope("session-a", "frame-3", 3))
            .unwrap();

        let stats = mailbox.stats();
        assert_eq!(stats.accepted_frames, 3);
        assert_eq!(stats.replaced_frames, 2);
        assert_eq!(stats.queue_depth, 1);
        assert_eq!(stats.max_queue_depth, 1);
        assert!(matches!(
            mailbox.receive(),
            ReceiveOutcome::Frame(frame) if frame.source.frame_id == "frame-3" && frame.rgb8[0] == 3
        ));
        assert_eq!(mailbox.stats().queue_depth, 0);
    }

    #[test]
    fn mailbox_rejects_frames_from_other_sessions_and_counts_them() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        let error = mailbox
            .publish(envelope("session-b", "foreign-frame", 0))
            .unwrap_err();
        assert_eq!(
            error,
            super::PublishError::SessionMismatch {
                expected: "session-a".to_owned(),
                actual: "session-b".to_owned()
            }
        );
        assert_eq!(mailbox.stats().rejected_frames, 1);
        assert_eq!(mailbox.stats().queue_depth, 0);
    }

    #[test]
    fn close_drains_the_final_frame_then_reports_closed() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        mailbox.publish(envelope("session-a", "last", 1)).unwrap();
        assert!(mailbox.close());
        assert!(!mailbox.close());
        assert!(matches!(
            mailbox.receive(),
            ReceiveOutcome::Frame(frame) if frame.source.frame_id == "last"
        ));
        assert_eq!(mailbox.receive(), ReceiveOutcome::Closed);
        assert_eq!(
            mailbox
                .publish(envelope("session-a", "late", 0))
                .unwrap_err(),
            super::PublishError::Terminal(MailboxTerminal::Closed)
        );
        assert_eq!(mailbox.stats().rejected_frames, 1);
    }

    #[test]
    fn cancel_discards_pending_frame_and_counts_cancellation_drop() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        mailbox
            .publish(envelope("session-a", "discard", 1))
            .unwrap();
        assert!(mailbox.cancel());
        assert!(!mailbox.cancel());
        assert_eq!(mailbox.receive(), ReceiveOutcome::Cancelled);
        let stats = mailbox.stats();
        assert_eq!(stats.cancelled_frames, 1);
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.max_queue_depth, 1);
    }

    #[test]
    fn producer_failure_drains_pending_frame_before_failure_outcome() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        mailbox
            .publish(envelope("session-a", "last-good", 1))
            .unwrap();
        assert!(mailbox.fail("device disconnected"));
        assert!(!mailbox.fail("second failure"));
        assert!(matches!(
            mailbox.receive(),
            ReceiveOutcome::Frame(frame) if frame.source.frame_id == "last-good"
        ));
        assert_eq!(
            mailbox.receive(),
            ReceiveOutcome::ProducerFailed("device disconnected".to_owned())
        );
    }

    #[test]
    fn timed_receive_reports_timeout_while_open_and_empty() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        assert_eq!(
            mailbox.receive_timeout(Duration::from_millis(10)),
            ReceiveOutcome::TimedOut
        );
    }

    #[test]
    fn waiting_consumer_wakes_when_a_frame_is_published() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        let consumer_mailbox = mailbox.clone();
        let consumer =
            thread::spawn(move || consumer_mailbox.receive_timeout(Duration::from_secs(2)));
        wait_for_receivers(&mailbox, 1, Duration::from_secs(1));
        assert_eq!(mailbox.stats().waiting_receivers, 1);
        mailbox.publish(envelope("session-a", "wake", 7)).unwrap();
        assert!(matches!(
            consumer.join().unwrap(),
            ReceiveOutcome::Frame(frame) if frame.source.frame_id == "wake"
        ));
        assert_eq!(mailbox.stats().waiting_receivers, 0);
    }

    #[test]
    fn producer_failure_wakes_an_empty_waiting_consumer() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        let consumer_mailbox = mailbox.clone();
        let consumer = thread::spawn(move || consumer_mailbox.receive());
        wait_for_receivers(&mailbox, 1, Duration::from_secs(1));
        assert_eq!(mailbox.stats().waiting_receivers, 1);
        mailbox.fail("capture stopped");
        assert_eq!(
            consumer.join().unwrap(),
            ReceiveOutcome::ProducerFailed("capture stopped".to_owned())
        );
        assert_eq!(mailbox.stats().waiting_receivers, 0);
    }

    #[test]
    fn receiver_claim_is_single_use_even_after_the_receiver_is_dropped() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        let receiver = mailbox.claim_receiver().expect("first claim succeeds");
        assert_eq!(receiver.session_id(), "session-a");
        assert_eq!(
            mailbox.claim_receiver().unwrap_err(),
            super::CaptureReceiverClaimError::AlreadyClaimed
        );

        drop(receiver);
        assert_eq!(
            mailbox.claim_receiver().unwrap_err(),
            super::CaptureReceiverClaimError::AlreadyClaimed
        );
    }

    #[test]
    fn synchronous_callback_orders_frames_while_slow_work_coalesces_pending_offers() {
        let mailbox = LatestFrameMailbox::new("session-a", TEST_FRAME_LIMIT).unwrap();
        let mut receiver = mailbox.claim_receiver().unwrap();
        mailbox
            .publish(envelope("session-a", "frame-1", 1))
            .unwrap();

        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let consumer = thread::spawn(move || {
            let first = receiver.receive_with_timeout(Duration::from_secs(2), |frame| {
                events_tx
                    .send(format!("start:{}", frame.source().frame_id))
                    .unwrap();
                resume_rx.recv().unwrap();
                events_tx
                    .send(format!("finish:{}", frame.source().frame_id))
                    .unwrap();
                frame.source().frame_id.clone()
            });
            let second = receiver.receive_with_timeout(Duration::from_secs(2), |frame| {
                events_tx
                    .send(format!("start:{}", frame.source().frame_id))
                    .unwrap();
                frame.source().frame_id.clone()
            });
            (first, second)
        });

        assert_eq!(
            events_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "start:frame-1"
        );
        mailbox
            .publish(envelope("session-a", "frame-2", 2))
            .unwrap();
        mailbox
            .publish(envelope("session-a", "frame-3", 3))
            .unwrap();
        assert_eq!(
            mailbox.stats(),
            super::MailboxStats {
                accepted_frames: 3,
                replaced_frames: 1,
                rejected_frames: 0,
                cancelled_frames: 0,
                queue_depth: 1,
                max_queue_depth: 1,
                waiting_receivers: 0,
            }
        );

        resume_tx.send(()).unwrap();
        assert_eq!(
            events_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "finish:frame-1"
        );
        assert_eq!(
            events_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "start:frame-3"
        );
        let (first, second) = consumer.join().unwrap();
        assert_eq!(
            first,
            super::CaptureReceiveOutcome::Frame("frame-1".to_owned())
        );
        assert_eq!(
            second,
            super::CaptureReceiveOutcome::Frame("frame-3".to_owned())
        );
    }

    #[test]
    fn receiver_reports_timeout_and_each_terminal_outcome_without_a_frame() {
        let timeout_mailbox = LatestFrameMailbox::new("session-timeout", TEST_FRAME_LIMIT).unwrap();
        let mut timeout_receiver = timeout_mailbox.claim_receiver().unwrap();
        assert_eq!(
            timeout_receiver.receive_with_timeout(Duration::ZERO, |_| unreachable!()),
            super::CaptureReceiveOutcome::TimedOut
        );

        let closed_mailbox = LatestFrameMailbox::new("session-closed", TEST_FRAME_LIMIT).unwrap();
        let mut closed_receiver = closed_mailbox.claim_receiver().unwrap();
        closed_mailbox.close();
        assert_eq!(
            closed_receiver.receive_with_timeout(Duration::from_secs(1), |_| unreachable!()),
            super::CaptureReceiveOutcome::Closed
        );

        let cancelled_mailbox =
            LatestFrameMailbox::new("session-cancelled", TEST_FRAME_LIMIT).unwrap();
        let mut cancelled_receiver = cancelled_mailbox.claim_receiver().unwrap();
        cancelled_mailbox.cancel();
        assert_eq!(
            cancelled_receiver.receive_with_timeout(Duration::from_secs(1), |_| unreachable!()),
            super::CaptureReceiveOutcome::Cancelled
        );

        let failed_mailbox = LatestFrameMailbox::new("session-failed", TEST_FRAME_LIMIT).unwrap();
        let mut failed_receiver = failed_mailbox.claim_receiver().unwrap();
        failed_mailbox.fail("camera disconnected");
        assert_eq!(
            failed_receiver.receive_with_timeout(Duration::from_secs(1), |_| unreachable!()),
            super::CaptureReceiveOutcome::ProducerFailed("camera disconnected".to_owned())
        );
    }

    #[test]
    fn receiver_delivers_the_final_pending_frame_before_graceful_close() {
        let mailbox = LatestFrameMailbox::new("session-final", TEST_FRAME_LIMIT).unwrap();
        let mut receiver = mailbox.claim_receiver().unwrap();
        mailbox
            .publish(envelope("session-final", "final-frame", 9))
            .unwrap();
        mailbox.close();

        assert_eq!(
            receiver.receive_with_timeout(Duration::from_secs(1), |frame| {
                frame.source().frame_id.clone()
            }),
            super::CaptureReceiveOutcome::Frame("final-frame".to_owned())
        );
        assert_eq!(
            receiver.receive_with_timeout(Duration::from_secs(1), |_| unreachable!()),
            super::CaptureReceiveOutcome::Closed
        );
    }
}

#[cfg(test)]
mod worker_tests {
    use super::{
        run_capture_worker, test_frame_budget, CaptureAggregateFrameLimit,
        CaptureAppliedCameraControls, CaptureAppliedFrameGeometry, CaptureCapability,
        CaptureControlReadback, CaptureExposureMode, CaptureFrameBudget, CaptureFrameLimit,
        CaptureFrameRate, CaptureInterrupt, CaptureOpenError, CaptureOwnerDisposition,
        CaptureOwnerGate, CaptureOwnerGateError, CaptureOwnerStatus, CapturePixelFormat,
        CaptureReadOutcome, CaptureRuntimeCoordinator, CaptureRuntimeWait, CaptureSessionReport,
        CaptureSource, CaptureSourceCapabilities, CaptureSourceFactory, CaptureSourceMetadata,
        CaptureStartError, CaptureTimestampSemantics, CaptureVideoProfile, CaptureWorker,
        CaptureWorkerEnd, CaptureWorkerFailure, CaptureWorkerInput, CaptureWorkerWait,
        LatestFrameMailbox, ReceiveOutcome, RgbFrameEnvelope, TEST_AGGREGATE_FRAME_LIMIT,
        TEST_FRAME_LIMIT,
    };
    use crate::detection_batch::{
        BatchTimestamp, ClockKind, SourceFrame, TimeUnit, TimestampStage,
    };
    use crate::frame_preview::LatestCapturedFrameSlot;
    use crate::live_config::{CaptureControlRequirements, ExpectedCaptureSettings};
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::thread::ThreadId;
    use std::time::{Duration, Instant};

    static RUNTIME_COORDINATOR_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Debug, Clone)]
    struct FakePlan {
        block_open: bool,
        interrupt_unblocks_open: bool,
        block_first_read: bool,
        interrupt_unblocks_read: bool,
        emit_frame_after_interrupt: bool,
        frame_count: usize,
        frame_id: String,
        open_error: Option<String>,
        open_panics: bool,
        metadata_panics: bool,
        read_error: Option<String>,
        read_panics: bool,
        block_release: bool,
        release_error: Option<String>,
        release_panics: bool,
        source_drop_panics: bool,
        metadata: CaptureSourceMetadata,
        frame_timestamp: Option<BatchTimestamp>,
    }

    impl Default for FakePlan {
        fn default() -> Self {
            Self {
                block_open: false,
                interrupt_unblocks_open: true,
                block_first_read: false,
                interrupt_unblocks_read: true,
                emit_frame_after_interrupt: false,
                frame_count: 0,
                frame_id: "frame-1".to_owned(),
                open_error: None,
                open_panics: false,
                metadata_panics: false,
                read_error: None,
                read_panics: false,
                block_release: false,
                release_error: None,
                release_panics: false,
                source_drop_panics: false,
                metadata: fake_metadata("RGB8"),
                frame_timestamp: Some(fake_source_timestamp("fake-clock")),
            }
        }
    }

    #[derive(Debug)]
    struct FakeState {
        plan: FakePlan,
        open_count: usize,
        metadata_count: usize,
        read_count: usize,
        frames_emitted: usize,
        release_count: usize,
        interrupt_called: bool,
        open_entered: bool,
        allow_read_finish: bool,
        read_entered: bool,
        release_entered: bool,
        allow_release_finish: bool,
        sessions_opened: Vec<String>,
        sessions_read: Vec<String>,
        owner_threads: Vec<ThreadId>,
        events: VecDeque<&'static str>,
    }

    #[derive(Debug)]
    struct FakeShared {
        state: Mutex<FakeState>,
        changed: Condvar,
    }

    impl FakeShared {
        fn new(plan: FakePlan) -> Arc<Self> {
            Arc::new(Self {
                state: Mutex::new(FakeState {
                    plan,
                    open_count: 0,
                    metadata_count: 0,
                    read_count: 0,
                    frames_emitted: 0,
                    release_count: 0,
                    interrupt_called: false,
                    open_entered: false,
                    allow_read_finish: false,
                    read_entered: false,
                    release_entered: false,
                    allow_release_finish: false,
                    sessions_opened: Vec::new(),
                    sessions_read: Vec::new(),
                    owner_threads: Vec::new(),
                    events: VecDeque::new(),
                }),
                changed: Condvar::new(),
            })
        }

        fn lock(&self) -> MutexGuard<'_, FakeState> {
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        fn wait_for(&self, timeout: Duration, predicate: impl Fn(&FakeState) -> bool) {
            let started_at = Instant::now();
            let mut state = self.lock();
            while !predicate(&state) {
                let remaining = timeout.saturating_sub(started_at.elapsed());
                assert!(
                    !remaining.is_zero(),
                    "fake-source event timed out: {state:?}"
                );
                let (next_state, wait) = self
                    .changed
                    .wait_timeout(state, remaining)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state = next_state;
                assert!(
                    !wait.timed_out() || predicate(&state),
                    "fake-source event timed out: {state:?}"
                );
            }
        }

        fn allow_read_to_finish(&self) {
            self.lock().allow_read_finish = true;
            self.changed.notify_all();
        }

        fn allow_release_to_finish(&self) {
            self.lock().allow_release_finish = true;
            self.changed.notify_all();
        }
    }

    #[derive(Debug)]
    struct FakeInterrupt(Arc<FakeShared>);

    impl CaptureInterrupt for FakeInterrupt {
        fn interrupt(&self) {
            self.0.lock().interrupt_called = true;
            self.0.changed.notify_all();
        }
    }

    #[derive(Debug)]
    struct FakeFactory(Arc<FakeShared>);

    impl CaptureSourceFactory for FakeFactory {
        fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
            Arc::new(FakeInterrupt(Arc::clone(&self.0)))
        }

        fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
            let shared = Arc::clone(&self.0);
            let plan = {
                let mut state = shared.lock();
                state.open_count += 1;
                state.open_entered = true;
                state.owner_threads.push(std::thread::current().id());
                state.events.push_back("open");
                let plan = state.plan.clone();
                shared.changed.notify_all();
                while plan.block_open && !(state.interrupt_called && plan.interrupt_unblocks_open) {
                    state = shared
                        .changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                let interrupted = state.interrupt_called;
                drop(state);
                (plan, interrupted)
            };
            let (plan, interrupted) = plan;
            if interrupted && plan.interrupt_unblocks_open {
                return Err(CaptureOpenError::Interrupted);
            }
            if plan.open_panics {
                panic!("fake open panic");
            }
            if let Some(message) = plan.open_error {
                // The factory contract requires cleanup of partial acquisition
                // before returning an open error.
                return Err(CaptureOpenError::Failed(message));
            }
            Ok(Box::new(FakeSource { shared, plan }))
        }

        fn open_with_session_id(
            self: Box<Self>,
            session_id: &str,
        ) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
            self.0.lock().sessions_opened.push(session_id.to_owned());
            self.open()
        }
    }

    #[derive(Debug)]
    struct FakeSource {
        shared: Arc<FakeShared>,
        plan: FakePlan,
    }

    impl CaptureSource for FakeSource {
        fn metadata(&self) -> CaptureSourceMetadata {
            let mut state = self.shared.lock();
            state.metadata_count += 1;
            state.owner_threads.push(std::thread::current().id());
            state.events.push_back("metadata");
            self.shared.changed.notify_all();
            if self.plan.metadata_panics {
                panic!("fake metadata panic");
            }
            self.plan.metadata.clone()
        }

        fn read(
            &mut self,
            session_id: &str,
            frame_limit: CaptureFrameLimit,
            frame_budget: &CaptureFrameBudget,
        ) -> Result<CaptureReadOutcome, String> {
            let mut state = self.shared.lock();
            state.read_count += 1;
            state.read_entered = true;
            state.sessions_read.push(session_id.to_owned());
            state.owner_threads.push(std::thread::current().id());
            state.events.push_back("read");
            self.shared.changed.notify_all();
            if self.plan.block_first_read && state.read_count == 1 {
                while !state.allow_read_finish
                    && !(state.interrupt_called && self.plan.interrupt_unblocks_read)
                {
                    state = self
                        .shared
                        .changed
                        .wait(state)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
            let interrupted = state.interrupt_called;
            let should_emit = state.frames_emitted < self.plan.frame_count
                && (!interrupted || self.plan.emit_frame_after_interrupt);
            if should_emit {
                state.frames_emitted += 1;
            }
            drop(state);

            if self.plan.read_panics {
                panic!("fake read panic");
            }
            if let Some(message) = &self.plan.read_error {
                return Err(message.clone());
            }
            if interrupted && !self.plan.emit_frame_after_interrupt {
                return Ok(CaptureReadOutcome::End);
            }
            if should_emit {
                return match fake_frame(
                    session_id,
                    &self.plan.frame_id,
                    self.plan.frame_timestamp.clone(),
                    frame_limit,
                    frame_budget,
                ) {
                    Ok(frame) => Ok(CaptureReadOutcome::Frame(frame)),
                    Err(error) => match error.drop_reason() {
                        Some(reason) => Ok(CaptureReadOutcome::Dropped(reason)),
                        None => Err(error.to_string()),
                    },
                };
            }
            Ok(CaptureReadOutcome::End)
        }

        fn release(&mut self) -> Result<(), String> {
            let mut state = self.shared.lock();
            state.release_count += 1;
            state.release_entered = true;
            state.owner_threads.push(std::thread::current().id());
            state.events.push_back("release-start");
            self.shared.changed.notify_all();
            while self.plan.block_release && !state.allow_release_finish {
                state = self
                    .shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            state.events.push_back("release-finish");
            drop(state);
            if self.plan.release_panics {
                panic!("fake release panic");
            }
            if let Some(message) = &self.plan.release_error {
                return Err(message.clone());
            }
            Ok(())
        }
    }

    impl Drop for FakeSource {
        fn drop(&mut self) {
            if self.plan.source_drop_panics {
                panic!("fake source destructor panic");
            }
        }
    }

    fn fake_frame(
        session_id: &str,
        frame_id: &str,
        timestamp: Option<BatchTimestamp>,
        frame_limit: CaptureFrameLimit,
        frame_budget: &CaptureFrameBudget,
    ) -> Result<RgbFrameEnvelope, super::CaptureFrameAllocationError> {
        let pixels = frame_budget.allocate_rgb8_buffer(2, 1, frame_limit)?;
        RgbFrameEnvelope::new(
            SourceFrame {
                session_id: session_id.to_owned(),
                frame_id: frame_id.to_owned(),
                timestamp,
                width_px: 2,
                height_px: 1,
            },
            pixels,
            Instant::now(),
            frame_limit,
        )
        .map_err(super::CaptureFrameAllocationError::InvalidFrame)
    }

    fn fake_metadata(pixel_format: &str) -> CaptureSourceMetadata {
        fake_metadata_with_timestamp_semantics(
            pixel_format,
            CaptureCapability::Supported,
            CaptureTimestampSemantics::SourceTimestamp {
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Sensor,
                domain_id: "fake-clock".to_owned(),
                stage: TimestampStage::Exposure,
            },
        )
    }

    #[test]
    fn compatible_capture_report_builds_request_bound_acknowledgement() {
        let metadata = fake_metadata("RGB8");
        let geometry = metadata.applied_frame_geometry();
        let expected = ExpectedCaptureSettings::new(
            geometry.configuration_revision(),
            metadata.device_id(),
            metadata.requested_profile().clone(),
            metadata.negotiated_profile().clone(),
            geometry.camera_profile_id(),
            geometry.image_space(),
            geometry.rectification_revision(),
            0,
            None,
            metadata.timestamp_semantics().clone(),
            CaptureControlRequirements::default(),
            Duration::from_secs(1),
            CaptureFrameLimit::new(6).unwrap(),
            CaptureAggregateFrameLimit::new(6).unwrap(),
        )
        .unwrap()
        .with_request_sha256("a".repeat(64));
        let report = CaptureSessionReport::new("capture-ack-test-session".to_owned(), metadata);

        let compatibility = expected.validate_session(&report).unwrap();
        let acknowledgement =
            crate::capture_session_ack::CaptureSessionAcknowledgementV1::from_compatible_session(
                &compatibility,
            )
            .unwrap();

        assert_eq!(acknowledgement.configuration_revision(), 1);
        assert_eq!(acknowledgement.request_sha256(), "a".repeat(64));
        assert_eq!(
            acknowledgement.session_id(),
            Some("capture-ack-test-session")
        );
        let serialized = serde_json::to_value(acknowledgement).unwrap();
        assert_eq!(
            serialized["source_report"],
            serde_json::to_value(report).unwrap()
        );
        assert_eq!(serialized["failure"], serde_json::Value::Null);
    }

    #[test]
    fn mailbox_publishes_shared_capture_frame_to_preview_before_inference_lags() {
        let session_id = "capture-preview-session";
        let metadata = fake_metadata("RGB8");
        let geometry = metadata.applied_frame_geometry();
        let expected = ExpectedCaptureSettings::new(
            geometry.configuration_revision(),
            metadata.device_id(),
            metadata.requested_profile().clone(),
            metadata.negotiated_profile().clone(),
            geometry.camera_profile_id(),
            geometry.image_space(),
            geometry.rectification_revision(),
            1,
            None,
            metadata.timestamp_semantics().clone(),
            CaptureControlRequirements::default(),
            Duration::from_secs(1),
            TEST_FRAME_LIMIT,
            TEST_AGGREGATE_FRAME_LIMIT,
        )
        .unwrap();
        let report = CaptureSessionReport::new(session_id.to_owned(), metadata);
        let compatibility = expected.validate_session(&report).unwrap();

        let budget = test_frame_budget();
        let preview = Arc::new(LatestCapturedFrameSlot::new("capture-preview-engine").unwrap());
        let mailbox = LatestFrameMailbox::with_budget_and_preview(
            session_id,
            TEST_FRAME_LIMIT,
            budget.clone(),
            Some(Arc::clone(&preview)),
        )
        .unwrap();
        mailbox.bind_frame_preview(&compatibility).unwrap();
        let make_frame = |frame_id: &str, fill: u8| {
            let mut pixels = budget.allocate_rgb8_buffer(2, 1, TEST_FRAME_LIMIT).unwrap();
            pixels.as_mut_slice().fill(fill);
            RgbFrameEnvelope::new(
                SourceFrame {
                    session_id: session_id.to_owned(),
                    frame_id: frame_id.to_owned(),
                    timestamp: Some(fake_source_timestamp("fake-clock")),
                    width_px: 2,
                    height_px: 1,
                },
                pixels,
                Instant::now(),
                TEST_FRAME_LIMIT,
            )
            .unwrap()
        };

        mailbox
            .publish(make_frame("frame-inference-is-processing", 19))
            .unwrap();
        let ReceiveOutcome::Frame(inference_frame) = mailbox.receive() else {
            panic!("first captured frame should be delivered to inference");
        };

        mailbox
            .publish(make_frame("frame-newest-capture", 221))
            .unwrap();
        let newest = preview.snapshot().expect("capture preview is current");
        assert_eq!(
            inference_frame.source().frame_id,
            "frame-inference-is-processing"
        );
        assert_eq!(newest.descriptor().source_frame_id, "frame-newest-capture");
        assert_eq!(
            newest.descriptor().engine_instance_id,
            "capture-preview-engine"
        );
        assert_eq!(newest.descriptor().capture_session_id, session_id);
        assert_eq!(newest.rgb8(), &[221; 6]);
        // Two captured buffers consume exactly two frame reservations: preview
        // shares the accepted envelope and does not allocate a pixel copy.
        assert_eq!(budget.stats().current_bytes, 12);

        mailbox.cancel();
        assert!(preview.snapshot().is_none());
        drop(newest);
        assert_eq!(budget.stats().current_bytes, 6);
        drop(inference_frame);
        assert_eq!(budget.stats().current_bytes, 0);
    }

    fn fake_metadata_read_completion_only(pixel_format: &str) -> CaptureSourceMetadata {
        fake_metadata_with_timestamp_semantics(
            pixel_format,
            CaptureCapability::Unknown,
            CaptureTimestampSemantics::ReadCompletionOnly,
        )
    }

    fn fake_metadata_with_timestamp_semantics(
        pixel_format: &str,
        timestamp_capability: CaptureCapability,
        timestamp_semantics: CaptureTimestampSemantics,
    ) -> CaptureSourceMetadata {
        let requested_format = CapturePixelFormat::new("RGB8").unwrap();
        let negotiated_format = CapturePixelFormat::new(pixel_format).unwrap();
        let requested = CaptureVideoProfile::new(
            2,
            1,
            CaptureFrameRate::new(30, 1).unwrap(),
            requested_format,
        )
        .unwrap();
        let negotiated = CaptureVideoProfile::new(
            2,
            1,
            CaptureFrameRate::new(30, 1).unwrap(),
            negotiated_format.clone(),
        )
        .unwrap();
        let capabilities = CaptureSourceCapabilities::new(
            Some(vec![negotiated_format]),
            CaptureCapability::Supported,
            CaptureCapability::Supported,
            CaptureCapability::Unknown,
            timestamp_capability,
        )
        .unwrap();
        CaptureSourceMetadata::new(
            "fake-camera",
            Some("Fake Camera".to_owned()),
            "fake-backend",
            requested,
            negotiated,
            capabilities,
            CaptureAppliedFrameGeometry::new(1, "fake-camera-profile", crate::ImageSpace::Raw, 0)
                .unwrap(),
            CaptureAppliedCameraControls::unknown(),
            timestamp_semantics,
        )
        .unwrap()
    }

    fn fake_source_timestamp(domain_id: &str) -> BatchTimestamp {
        BatchTimestamp {
            ticks: "7".to_owned(),
            unit: TimeUnit::Nanoseconds,
            clock: ClockKind::Sensor,
            domain_id: domain_id.to_owned(),
            stage: TimestampStage::Exposure,
        }
    }

    #[test]
    fn capture_metadata_rejects_invalid_profiles_and_capability_claims() {
        assert_eq!(
            CaptureFrameRate::new(0, 1),
            Err(super::CaptureMetadataError::ZeroFrameRateNumerator)
        );
        assert_eq!(
            CaptureFrameRate::new(30, 0),
            Err(super::CaptureMetadataError::ZeroFrameRateDenominator)
        );
        assert_eq!(
            CapturePixelFormat::new(""),
            Err(super::CaptureMetadataError::InvalidPixelFormat(
                String::new()
            ))
        );
        assert_eq!(
            CaptureAppliedFrameGeometry::new(0, "camera-profile", crate::ImageSpace::Raw, 0,),
            Err(super::CaptureMetadataError::InvalidRevision(
                "source.configuration_revision"
            ))
        );
        assert_eq!(
            CaptureAppliedFrameGeometry::new(
                1,
                "camera-profile",
                crate::ImageSpace::Raw,
                9_007_199_254_740_992,
            ),
            Err(super::CaptureMetadataError::InvalidRevision(
                "source.rectification_revision"
            ))
        );
        assert!(matches!(
            CaptureAppliedFrameGeometry::new(1, "camera profile", crate::ImageSpace::Raw, 0,),
            Err(super::CaptureMetadataError::InvalidIdentifier {
                field: "source.camera_profile_id"
            })
        ));

        let frame_rate = CaptureFrameRate::new(30, 1).unwrap();
        let pixel_format = CapturePixelFormat::new("RGB8").unwrap();
        assert_eq!(
            CaptureVideoProfile::new(0, 1, frame_rate, pixel_format.clone()),
            Err(super::CaptureMetadataError::ZeroWidth)
        );
        assert_eq!(
            CaptureVideoProfile::new(1, 0, frame_rate, pixel_format.clone()),
            Err(super::CaptureMetadataError::ZeroHeight)
        );

        assert_eq!(
            CaptureSourceCapabilities::new(
                Some(Vec::new()),
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
            ),
            Err(super::CaptureMetadataError::EmptySupportedPixelFormats)
        );
        assert_eq!(
            CaptureSourceCapabilities::new(
                Some(vec![pixel_format.clone(), pixel_format.clone()]),
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
            ),
            Err(super::CaptureMetadataError::DuplicateSupportedPixelFormat(
                "RGB8".to_owned()
            ))
        );

        let requested =
            CaptureVideoProfile::new(2, 1, frame_rate, CapturePixelFormat::new("RGB8").unwrap())
                .unwrap();
        let negotiated =
            CaptureVideoProfile::new(2, 1, frame_rate, CapturePixelFormat::new("YUYV").unwrap())
                .unwrap();
        let capabilities = CaptureSourceCapabilities::new(
            Some(vec![CapturePixelFormat::new("RGB8").unwrap()]),
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
        )
        .unwrap();
        assert_eq!(
            CaptureSourceMetadata::new(
                "camera",
                None,
                "backend",
                requested,
                negotiated,
                capabilities,
                CaptureAppliedFrameGeometry::new(
                    1,
                    "test-camera-profile",
                    crate::ImageSpace::Raw,
                    0,
                )
                .unwrap(),
                CaptureAppliedCameraControls::unknown(),
                CaptureTimestampSemantics::ReadCompletionOnly,
            ),
            Err(super::CaptureMetadataError::NegotiatedPixelFormatNotSupported("YUYV".to_owned()))
        );

        let valid_requested =
            CaptureVideoProfile::new(2, 1, frame_rate, pixel_format.clone()).unwrap();
        let valid_negotiated =
            CaptureVideoProfile::new(2, 1, frame_rate, pixel_format.clone()).unwrap();
        let valid_capabilities = CaptureSourceCapabilities::new(
            Some(vec![pixel_format]),
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
        )
        .unwrap();
        assert_eq!(
            CaptureSourceMetadata::new(
                "",
                None,
                "backend",
                valid_requested.clone(),
                valid_negotiated.clone(),
                valid_capabilities.clone(),
                CaptureAppliedFrameGeometry::new(
                    1,
                    "test-camera-profile",
                    crate::ImageSpace::Raw,
                    0,
                )
                .unwrap(),
                CaptureAppliedCameraControls::unknown(),
                CaptureTimestampSemantics::ReadCompletionOnly,
            ),
            Err(super::CaptureMetadataError::InvalidIdentifier {
                field: "source.device_id"
            })
        );
        assert_eq!(
            CaptureSourceMetadata::new(
                "camera",
                Some(String::new()),
                "backend",
                valid_requested.clone(),
                valid_negotiated.clone(),
                valid_capabilities.clone(),
                CaptureAppliedFrameGeometry::new(
                    1,
                    "test-camera-profile",
                    crate::ImageSpace::Raw,
                    0,
                )
                .unwrap(),
                CaptureAppliedCameraControls::unknown(),
                CaptureTimestampSemantics::ReadCompletionOnly,
            ),
            Err(super::CaptureMetadataError::InvalidIdentifier {
                field: "source.device_name"
            })
        );
        assert!(CaptureSourceMetadata::new(
            "camera",
            None,
            "backend",
            valid_requested.clone(),
            valid_negotiated.clone(),
            valid_capabilities.clone(),
            CaptureAppliedFrameGeometry::new(1, "test-camera-profile", crate::ImageSpace::Raw, 0,)
                .unwrap(),
            CaptureAppliedCameraControls::unknown(),
            CaptureTimestampSemantics::ReadCompletionOnly,
        )
        .is_ok());

        for capability in [CaptureCapability::Unsupported, CaptureCapability::Unknown] {
            let capabilities = CaptureSourceCapabilities::new(
                Some(vec![CapturePixelFormat::new("RGB8").unwrap()]),
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
                CaptureCapability::Unknown,
                capability,
            )
            .unwrap();
            assert_eq!(
                CaptureSourceMetadata::new(
                    "camera",
                    None,
                    "backend",
                    valid_requested.clone(),
                    valid_negotiated.clone(),
                    capabilities,
                    CaptureAppliedFrameGeometry::new(
                        1,
                        "test-camera-profile",
                        crate::ImageSpace::Raw,
                        0,
                    )
                    .unwrap(),
                    CaptureAppliedCameraControls::unknown(),
                    CaptureTimestampSemantics::SourceTimestamp {
                        unit: TimeUnit::Nanoseconds,
                        clock: ClockKind::Sensor,
                        domain_id: "sensor-clock".to_owned(),
                        stage: TimestampStage::Exposure,
                    },
                ),
                Err(super::CaptureMetadataError::TimestampCapabilityMismatch {
                    capability,
                    reports_source_timestamp: true,
                })
            );
        }

        let timestamped_capabilities = CaptureSourceCapabilities::new(
            Some(vec![CapturePixelFormat::new("RGB8").unwrap()]),
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
            CaptureCapability::Unknown,
            CaptureCapability::Supported,
        )
        .unwrap();
        assert_eq!(
            CaptureSourceMetadata::new(
                "camera",
                None,
                "backend",
                valid_requested.clone(),
                valid_negotiated.clone(),
                timestamped_capabilities,
                CaptureAppliedFrameGeometry::new(
                    1,
                    "test-camera-profile",
                    crate::ImageSpace::Raw,
                    0,
                )
                .unwrap(),
                CaptureAppliedCameraControls::unknown(),
                CaptureTimestampSemantics::ReadCompletionOnly,
            ),
            Err(super::CaptureMetadataError::TimestampCapabilityMismatch {
                capability: CaptureCapability::Supported,
                reports_source_timestamp: false,
            })
        );
        assert_eq!(
            CaptureSourceMetadata::new(
                "camera",
                None,
                "backend",
                valid_requested,
                valid_negotiated,
                valid_capabilities,
                CaptureAppliedFrameGeometry::new(
                    1,
                    "test-camera-profile",
                    crate::ImageSpace::Raw,
                    0,
                )
                .unwrap(),
                CaptureAppliedCameraControls::unknown(),
                CaptureTimestampSemantics::SourceTimestamp {
                    unit: TimeUnit::Nanoseconds,
                    clock: ClockKind::Sensor,
                    domain_id: "sensor-clock".to_owned(),
                    stage: TimestampStage::InferenceComplete,
                },
            ),
            Err(super::CaptureMetadataError::InvalidTimestampStage(
                TimestampStage::InferenceComplete
            ))
        );
    }

    #[test]
    fn applied_control_readbacks_keep_unknown_states_separate_from_values() {
        let applied = CaptureAppliedCameraControls::new(
            CaptureControlReadback::Applied(CaptureExposureMode::Automatic),
            CaptureControlReadback::Applied(-4.0),
            CaptureControlReadback::Applied(1.25),
            CaptureControlReadback::Unsupported,
            CaptureControlReadback::Unknown,
            CaptureControlReadback::Applied(0.75),
            CaptureControlReadback::Applied(true),
            CaptureControlReadback::Applied(2),
        )
        .unwrap();
        let serialized = serde_json::to_value(&applied).unwrap();
        assert_eq!(
            serialized["exposure_mode"],
            serde_json::json!({"status": "applied", "value": "automatic"})
        );
        assert_eq!(
            serialized["exposure_value"],
            serde_json::json!({"status": "applied", "value": -4.0})
        );
        assert_eq!(
            serialized["gain"],
            serde_json::json!({"status": "applied", "value": 1.25})
        );
        assert_eq!(
            serialized["brightness"],
            serde_json::json!({"status": "unsupported"})
        );
        assert_eq!(
            serialized["contrast"],
            serde_json::json!({"status": "unknown"})
        );
        assert_eq!(serialized["auto_focus"]["value"], true);
        assert_eq!(serialized["buffer_size"]["value"], 2);

        assert_eq!(
            CaptureAppliedCameraControls::new(
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Applied(f64::NAN),
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
            ),
            Err(super::CaptureMetadataError::InvalidAppliedControl(
                "exposure_value"
            ))
        );
        assert_eq!(
            CaptureAppliedCameraControls::new(
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Applied(-0.1),
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
            ),
            Err(super::CaptureMetadataError::InvalidAppliedControl("gain"))
        );
    }

    #[test]
    fn coordinator_exposes_metadata_only_for_a_successfully_opened_source() {
        let _serial = RUNTIME_COORDINATOR_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();

        let pending_open = FakeShared::new(FakePlan {
            block_open: true,
            ..FakePlan::default()
        });
        runtime
            .start(FakeFactory(Arc::clone(&pending_open)))
            .expect("pending source starts");
        pending_open.wait_for(Duration::from_secs(1), |state| state.open_entered);
        assert_eq!(runtime.active_source_report(), None);
        assert!(matches!(
            runtime.stop(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
        assert_eq!(runtime.active_source_report(), None);

        let failed_open = FakeShared::new(FakePlan {
            open_error: Some("device unavailable".to_owned()),
            ..FakePlan::default()
        });
        runtime
            .start(FakeFactory(Arc::clone(&failed_open)))
            .expect("failed open still starts its worker");
        let failed = match runtime.wait_timeout(Duration::from_secs(1)) {
            CaptureRuntimeWait::Completed(report) => report,
            other => panic!("failed open did not complete: {other:?}"),
        };
        assert_eq!(failed.end, CaptureWorkerEnd::Failed);
        assert_eq!(runtime.active_source_report(), None);

        let ready_source = FakeShared::new(FakePlan {
            block_first_read: true,
            ..FakePlan::default()
        });
        runtime
            .start(FakeFactory(Arc::clone(&ready_source)))
            .expect("ready source starts");
        ready_source.wait_for(Duration::from_secs(1), |state| state.read_entered);
        let report = runtime
            .active_source_report()
            .expect("opened source metadata is available");
        assert_eq!(
            ready_source.lock().sessions_opened,
            vec![report.session_id().to_owned()]
        );
        assert_eq!(report.session_id(), ready_source.lock().sessions_read[0]);
        assert_eq!(
            report.negotiated_profile().pixel_format().identifier(),
            "RGB8"
        );
        assert_eq!(report.device_id(), "fake-camera");
        assert_eq!(report.backend_id(), "fake-backend");
        assert_eq!(ready_source.lock().metadata_count, 1);
        assert!(matches!(
            runtime.stop(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
        assert_eq!(runtime.active_source_report(), None);
    }

    #[test]
    fn restarted_coordinator_reports_metadata_for_the_new_session_only() {
        let _serial = RUNTIME_COORDINATOR_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();

        let first_source = FakeShared::new(FakePlan {
            block_first_read: true,
            metadata: fake_metadata("RGB8"),
            ..FakePlan::default()
        });
        let first_session = runtime
            .start(FakeFactory(Arc::clone(&first_source)))
            .expect("first source starts");
        first_source.wait_for(Duration::from_secs(1), |state| state.read_entered);
        let first_report = runtime
            .active_source_report()
            .expect("first opened session is ready");
        assert_eq!(first_report.session_id(), first_session);
        assert_eq!(first_report.device_name(), Some("Fake Camera"));
        assert_eq!(
            first_report
                .negotiated_profile()
                .pixel_format()
                .identifier(),
            "RGB8"
        );
        assert_eq!(
            first_report.requested_profile().pixel_format().identifier(),
            "RGB8"
        );
        assert_eq!(
            first_report.timestamp_semantics(),
            &CaptureTimestampSemantics::SourceTimestamp {
                unit: TimeUnit::Nanoseconds,
                clock: ClockKind::Sensor,
                domain_id: "fake-clock".to_owned(),
                stage: TimestampStage::Exposure,
            }
        );
        assert!(matches!(
            runtime.stop(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
        assert_eq!(runtime.active_source_report(), None);

        let second_source = FakeShared::new(FakePlan {
            block_first_read: true,
            metadata: fake_metadata("YUYV"),
            ..FakePlan::default()
        });
        let second_session = runtime
            .start(FakeFactory(Arc::clone(&second_source)))
            .expect("replacement source starts after release");
        second_source.wait_for(Duration::from_secs(1), |state| state.read_entered);
        let second_report = runtime
            .active_source_report()
            .expect("replacement opened session is ready");
        assert_ne!(first_session, second_session);
        assert_eq!(second_report.session_id(), second_session);
        assert_eq!(
            second_report
                .negotiated_profile()
                .pixel_format()
                .identifier(),
            "YUYV"
        );
        let serialized = serde_json::to_value(&second_report).unwrap();
        assert_eq!(serialized["session_id"], second_session);
        assert_eq!(serialized["device_id"], "fake-camera");
        assert_eq!(serialized["device_name"], "Fake Camera");
        assert_eq!(serialized["backend_id"], "fake-backend");
        assert_eq!(serialized["requested_profile"]["pixel_format"], "RGB8");
        assert_eq!(serialized["negotiated_profile"]["pixel_format"], "YUYV");
        assert_eq!(
            serialized["timestamp_semantics"]["kind"],
            "source_timestamp"
        );
        assert!(matches!(
            runtime.stop(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
    }

    #[test]
    fn source_metadata_read_keeps_capture_worker_on_one_owner_thread() {
        let shared = FakeShared::new(FakePlan {
            frame_count: 1,
            ..FakePlan::default()
        });
        let mut worker = CaptureWorker::start_for_test(
            CaptureOwnerGate::new(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        let mut receiver = worker.claim_receiver().expect("receiver claimed");
        let report = completed(worker.wait_timeout(Duration::from_secs(1)));
        assert_eq!(report.end, CaptureWorkerEnd::NaturalEnd);
        assert_eq!(report.frames_published, 1);
        assert_eq!(shared.lock().metadata_count, 1);
        assert_owner_thread_for_all_source_calls(&shared);
        assert_eq!(
            receiver.receive_with_timeout(Duration::from_secs(1), |frame| {
                frame.source().frame_id.clone()
            }),
            super::CaptureReceiveOutcome::Frame("frame-1".to_owned())
        );
    }

    #[test]
    fn capture_worker_rejects_frames_that_conflict_with_reported_timestamp_semantics() {
        let source_timestamp_semantics = CaptureTimestampSemantics::SourceTimestamp {
            unit: TimeUnit::Nanoseconds,
            clock: ClockKind::Sensor,
            domain_id: "fake-clock".to_owned(),
            stage: TimestampStage::Exposure,
        };
        let wrong_domain_timestamp = fake_source_timestamp("other-clock");
        let wrong_clock_timestamp = BatchTimestamp {
            ticks: "7".to_owned(),
            unit: TimeUnit::Nanoseconds,
            clock: ClockKind::Monotonic,
            domain_id: "fake-clock".to_owned(),
            stage: TimestampStage::Exposure,
        };
        let wrong_stage_timestamp = BatchTimestamp {
            ticks: "7".to_owned(),
            unit: TimeUnit::Nanoseconds,
            clock: ClockKind::Sensor,
            domain_id: "fake-clock".to_owned(),
            stage: TimestampStage::CameraReadComplete,
        };
        let cases = [
            (
                fake_metadata("RGB8"),
                None,
                source_timestamp_semantics.clone(),
                None,
            ),
            (
                fake_metadata_read_completion_only("RGB8"),
                Some(fake_source_timestamp("fake-clock")),
                CaptureTimestampSemantics::ReadCompletionOnly,
                Some(CaptureTimestampSemantics::SourceTimestamp {
                    unit: TimeUnit::Nanoseconds,
                    clock: ClockKind::Sensor,
                    domain_id: "fake-clock".to_owned(),
                    stage: TimestampStage::Exposure,
                }),
            ),
            (
                fake_metadata("RGB8"),
                Some(wrong_domain_timestamp.clone()),
                source_timestamp_semantics.clone(),
                Some(CaptureTimestampSemantics::SourceTimestamp {
                    unit: wrong_domain_timestamp.unit,
                    clock: wrong_domain_timestamp.clock,
                    domain_id: wrong_domain_timestamp.domain_id.clone(),
                    stage: wrong_domain_timestamp.stage,
                }),
            ),
            (
                fake_metadata("RGB8"),
                Some(wrong_clock_timestamp.clone()),
                source_timestamp_semantics.clone(),
                Some(CaptureTimestampSemantics::SourceTimestamp {
                    unit: wrong_clock_timestamp.unit,
                    clock: wrong_clock_timestamp.clock,
                    domain_id: wrong_clock_timestamp.domain_id.clone(),
                    stage: wrong_clock_timestamp.stage,
                }),
            ),
            (
                fake_metadata("RGB8"),
                Some(wrong_stage_timestamp.clone()),
                source_timestamp_semantics.clone(),
                Some(CaptureTimestampSemantics::SourceTimestamp {
                    unit: wrong_stage_timestamp.unit,
                    clock: wrong_stage_timestamp.clock,
                    domain_id: wrong_stage_timestamp.domain_id.clone(),
                    stage: wrong_stage_timestamp.stage,
                }),
            ),
        ];

        for (metadata, frame_timestamp, expected_semantics, expected_timestamp) in cases {
            let shared = FakeShared::new(FakePlan {
                frame_count: 1,
                metadata,
                frame_timestamp,
                ..FakePlan::default()
            });
            let mut worker = CaptureWorker::start_for_test(
                CaptureOwnerGate::new(),
                TEST_FRAME_LIMIT,
                FakeFactory(Arc::clone(&shared)),
            )
            .expect("worker starts");
            let mut receiver = worker.claim_receiver().expect("receiver claimed");
            let report = completed(worker.wait_timeout(Duration::from_secs(1)));

            assert_eq!(report.end, CaptureWorkerEnd::Failed);
            assert_eq!(report.frames_published, 0);
            assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
            assert_eq!(
                report.failures,
                vec![CaptureWorkerFailure::FrameTimestampMismatch {
                    expected: expected_semantics,
                    actual: expected_timestamp,
                }]
            );
            assert_eq!(shared.lock().release_count, 1);
            assert_eq!(
                receiver.receive_with_timeout(Duration::from_secs(1), |_| ()),
                super::CaptureReceiveOutcome::ProducerFailed(report.failures[0].to_string())
            );
        }
    }

    #[test]
    fn capture_worker_preserves_read_completion_only_when_frame_has_no_source_time() {
        let shared = FakeShared::new(FakePlan {
            frame_count: 1,
            metadata: fake_metadata_read_completion_only("RGB8"),
            frame_timestamp: None,
            ..FakePlan::default()
        });
        let mut worker = CaptureWorker::start_for_test(
            CaptureOwnerGate::new(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        let mut receiver = worker.claim_receiver().expect("receiver claimed");
        let report = completed(worker.wait_timeout(Duration::from_secs(1)));

        assert_eq!(report.end, CaptureWorkerEnd::NaturalEnd);
        assert_eq!(report.frames_published, 1);
        assert!(report.failures.is_empty());
        assert_eq!(
            receiver.receive_with_timeout(Duration::from_secs(1), |frame| {
                frame.source().timestamp.clone()
            }),
            super::CaptureReceiveOutcome::Frame(None)
        );
    }

    #[test]
    fn metadata_panic_releases_open_source_without_publishing_ready_report() {
        let _serial = RUNTIME_COORDINATOR_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let shared = FakeShared::new(FakePlan {
            metadata_panics: true,
            ..FakePlan::default()
        });
        let mut runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();
        runtime
            .start(FakeFactory(Arc::clone(&shared)))
            .expect("source worker starts");
        let report = match runtime.wait_timeout(Duration::from_secs(1)) {
            CaptureRuntimeWait::Completed(report) => report,
            other => panic!("metadata panic did not complete the worker: {other:?}"),
        };
        assert_eq!(report.end, CaptureWorkerEnd::Failed);
        assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            CaptureWorkerFailure::SourceMetadataPanicked(message)
                if message == "fake metadata panic"
        )));
        assert_eq!(shared.lock().release_count, 1);
        assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
        assert_eq!(runtime.active_source_report(), None);
    }

    fn completed(report: CaptureWorkerWait) -> super::CaptureWorkerReport {
        match report {
            CaptureWorkerWait::Completed(report) => report,
            CaptureWorkerWait::TimedOut => panic!("capture worker unexpectedly timed out"),
        }
    }

    fn assert_owner_thread_for_all_source_calls(shared: &FakeShared) {
        let state = shared.lock();
        assert_eq!(
            state.owner_threads.len(),
            state.open_count + state.metadata_count + state.read_count + state.release_count
        );
        assert!(!state.owner_threads.is_empty());
        assert!(state
            .owner_threads
            .iter()
            .all(|thread_id| *thread_id == state.owner_threads[0]));
        assert_eq!(state.events.front(), Some(&"open"));
        assert_eq!(state.events.back(), Some(&"release-finish"));
    }

    #[test]
    fn worker_continues_after_budget_drop_and_reports_it_without_failure() {
        let budget = CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(6).unwrap());
        let plan = FakePlan {
            frame_count: 2,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let mut worker = CaptureWorker::start(
            gate,
            TEST_FRAME_LIMIT,
            budget.clone(),
            FakeFactory(Arc::clone(&shared)),
            None,
            None,
        )
        .expect("worker starts");
        let mut receiver = worker.claim_receiver().expect("sole receiver claimed");

        let report = completed(worker.wait_timeout(Duration::from_secs(1)));
        assert_eq!(report.end, CaptureWorkerEnd::NaturalEnd);
        assert_eq!(report.frames_published, 1);
        assert_eq!(report.frames_dropped, 1);
        assert_eq!(report.budget_exhausted_frames, 1);
        assert!(report.failures.is_empty());
        assert_eq!(shared.lock().read_count, 3);
        assert_eq!(budget.stats().current_bytes, 6);

        assert_eq!(
            receiver.receive_with_timeout(Duration::from_secs(1), |frame| {
                frame.source().frame_id.clone()
            }),
            super::CaptureReceiveOutcome::Frame("frame-1".to_owned())
        );
        assert_eq!(budget.stats().current_bytes, 0);
    }

    #[test]
    fn coordinators_share_old_envelope_reservations_and_reject_conflicting_limits() {
        let _serial = RUNTIME_COORDINATOR_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();
        let old_envelope = super::tests::envelope_from_budget(
            &first.frame_budget,
            "session-old",
            "frame-old",
            0,
            2,
            1,
            TEST_FRAME_LIMIT,
        )
        .unwrap();
        assert_eq!(first.frame_budget_stats().current_bytes, 6);
        drop(first);

        let replacement =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();
        assert_eq!(replacement.frame_budget_stats().current_bytes, 6);
        drop(old_envelope);
        assert_eq!(replacement.frame_budget_stats().current_bytes, 0);
        assert!(matches!(
            CaptureRuntimeCoordinator::new(
                TEST_FRAME_LIMIT,
                CaptureAggregateFrameLimit::new(
                    TEST_AGGREGATE_FRAME_LIMIT.maximum_rgb8_bytes() + 1
                )
                .unwrap()
            ),
            Err(super::CaptureRuntimeConfigurationError::ConflictingAggregateFrameLimit {
                configured_bytes,
                requested_bytes
            }) if configured_bytes == TEST_AGGREGATE_FRAME_LIMIT.maximum_rgb8_bytes()
                && requested_bytes == TEST_AGGREGATE_FRAME_LIMIT.maximum_rgb8_bytes() + 1
        ));
    }

    #[test]
    fn runtime_coordinator_keeps_gate_private_and_serializes_source_starts() {
        let _serial = RUNTIME_COORDINATOR_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first_plan = FakePlan {
            block_open: true,
            interrupt_unblocks_open: true,
            ..FakePlan::default()
        };
        let first_shared = FakeShared::new(first_plan);
        let second_shared = FakeShared::new(FakePlan::default());
        let mut runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();
        let mut second_runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();

        runtime
            .start(FakeFactory(Arc::clone(&first_shared)))
            .expect("first source starts");
        first_shared.wait_for(Duration::from_secs(1), |state| state.open_entered);
        assert!(matches!(
            runtime.start(FakeFactory(Arc::clone(&second_shared))),
            Err(CaptureStartError::WorkerStillRunning { .. })
        ));
        assert!(matches!(
            second_runtime.start(FakeFactory(Arc::clone(&second_shared))),
            Err(CaptureStartError::OwnerGate(
                CaptureOwnerGateError::AlreadyOwned { .. }
            ))
        ));
        assert_eq!(second_shared.lock().open_count, 0);
        assert!(matches!(
            runtime.stop(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
        assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);

        second_runtime
            .start(FakeFactory(Arc::clone(&second_shared)))
            .expect("source may restart after confirmed release");
        assert!(matches!(
            second_runtime.wait_timeout(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
        assert_eq!(second_shared.lock().open_count, 1);
    }

    #[test]
    fn runtime_coordinator_claims_one_receiver_and_preserves_timeout_and_cancel_outcomes() {
        let _serial = RUNTIME_COORDINATOR_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let plan = FakePlan {
            block_first_read: true,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let mut runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();
        assert_eq!(
            runtime.claim_receiver().unwrap_err(),
            super::CaptureReceiverClaimError::NoActiveSession
        );

        runtime
            .start(FakeFactory(Arc::clone(&shared)))
            .expect("capture starts");
        shared.wait_for(Duration::from_secs(1), |state| state.read_entered);
        let mut receiver = runtime
            .claim_receiver()
            .expect("active session grants its sole receiver");
        assert_eq!(
            runtime.claim_receiver().unwrap_err(),
            super::CaptureReceiverClaimError::AlreadyClaimed
        );
        assert_eq!(
            receiver.receive_with_timeout(Duration::from_millis(5), |_| ()),
            super::CaptureReceiveOutcome::TimedOut
        );

        assert!(matches!(
            runtime.stop(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
        assert_eq!(
            receiver.receive_with_timeout(Duration::from_secs(1), |_| ()),
            super::CaptureReceiveOutcome::Cancelled
        );
        assert_eq!(
            runtime.claim_receiver().unwrap_err(),
            super::CaptureReceiverClaimError::NoActiveSession
        );
    }

    #[test]
    fn runtime_coordinator_retains_timed_out_worker_and_rejects_replacement() {
        let _serial = RUNTIME_COORDINATOR_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let blocked_read = FakePlan {
            block_first_read: true,
            interrupt_unblocks_read: false,
            frame_count: 1,
            ..FakePlan::default()
        };
        let first_shared = FakeShared::new(blocked_read);
        let second_shared = FakeShared::new(FakePlan::default());
        let mut runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();
        let mut replacement_runtime =
            CaptureRuntimeCoordinator::new(TEST_FRAME_LIMIT, TEST_AGGREGATE_FRAME_LIMIT).unwrap();
        runtime
            .start(FakeFactory(Arc::clone(&first_shared)))
            .expect("first source starts");
        first_shared.wait_for(Duration::from_secs(1), |state| state.read_entered);

        assert_eq!(
            runtime.stop(Duration::from_millis(1)),
            CaptureRuntimeWait::TimedOut
        );
        assert!(matches!(
            replacement_runtime.start(FakeFactory(Arc::clone(&second_shared))),
            Err(CaptureStartError::OwnerGate(
                CaptureOwnerGateError::AlreadyOwned { .. }
            ))
        ));
        assert_eq!(second_shared.lock().open_count, 0);

        first_shared.allow_read_to_finish();
        assert!(matches!(
            runtime.wait_timeout(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
        assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
        replacement_runtime
            .start(FakeFactory(Arc::clone(&second_shared)))
            .expect("replacement starts after old worker exits");
        assert!(matches!(
            replacement_runtime.wait_timeout(Duration::from_secs(1)),
            CaptureRuntimeWait::Completed(_)
        ));
    }

    #[test]
    fn stop_already_requested_before_open_skips_factory_and_cancels_mailbox() {
        let shared = FakeShared::new(FakePlan::default());
        let session_id = "session-pre-cancel".to_owned();
        let mailbox = LatestFrameMailbox::new(session_id.clone(), TEST_FRAME_LIMIT).unwrap();
        let report = run_capture_worker(CaptureWorkerInput {
            factory: Box::new(FakeFactory(Arc::clone(&shared))),
            session_id,
            frame_limit: TEST_FRAME_LIMIT,
            frame_budget: test_frame_budget(),
            mailbox: mailbox.clone(),
            stop_requested: Arc::new(AtomicBool::new(true)),
            ready_report: Arc::new(Mutex::new(None)),
            preflight: None,
        });

        assert_eq!(report.end, CaptureWorkerEnd::Stopped);
        assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
        assert!(report.failures.is_empty());
        assert_eq!(shared.lock().open_count, 0);
        assert_eq!(mailbox.receive(), ReceiveOutcome::Cancelled);
    }

    #[test]
    fn stop_interrupts_blocked_open_before_any_source_read() {
        let plan = FakePlan {
            block_open: true,
            interrupt_unblocks_open: true,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let mut worker = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        let mailbox = worker.mailbox();
        shared.wait_for(Duration::from_secs(1), |state| state.open_entered);

        let report = completed(worker.stop(Duration::from_secs(1)));
        assert_eq!(report.end, CaptureWorkerEnd::Stopped);
        assert!(report.failures.is_empty());
        assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
        assert_eq!(shared.lock().read_count, 0);
        assert_eq!(shared.lock().release_count, 0);
        assert_eq!(mailbox.receive(), ReceiveOutcome::Cancelled);
        assert_eq!(gate.status(), CaptureOwnerStatus::Available);
    }

    #[test]
    fn interrupted_blocking_read_releases_on_owner_thread_before_completion() {
        let plan = FakePlan {
            block_first_read: true,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let mut worker = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        shared.wait_for(Duration::from_secs(1), |state| state.read_entered);

        let report = completed(worker.stop(Duration::from_secs(1)));
        assert_eq!(report.end, CaptureWorkerEnd::Stopped);
        assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
        assert!(report.failures.is_empty());
        assert_eq!(shared.lock().release_count, 1);
        assert_eq!(worker.mailbox().receive(), ReceiveOutcome::Cancelled);
        assert_eq!(gate.status(), CaptureOwnerStatus::Available);
        assert_owner_thread_for_all_source_calls(&shared);
    }

    #[test]
    fn timeout_retains_worker_gate_and_prevents_open_until_late_read_is_released() {
        let plan = FakePlan {
            block_first_read: true,
            interrupt_unblocks_read: false,
            emit_frame_after_interrupt: true,
            frame_count: 1,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let mut worker = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("first worker starts");
        let mailbox: LatestFrameMailbox = worker.mailbox();
        shared.wait_for(Duration::from_secs(1), |state| state.read_entered);

        assert_eq!(
            worker.stop(Duration::from_millis(5)),
            CaptureWorkerWait::TimedOut
        );
        assert!(matches!(gate.status(), CaptureOwnerStatus::Owned { .. }));
        let second_shared = FakeShared::new(FakePlan::default());
        assert!(matches!(
            CaptureWorker::start_for_test(
                gate.clone(),
                TEST_FRAME_LIMIT,
                FakeFactory(Arc::clone(&second_shared))
            ),
            Err(CaptureStartError::OwnerGate(
                CaptureOwnerGateError::AlreadyOwned { .. }
            ))
        ));
        assert_eq!(second_shared.lock().open_count, 0);

        shared.allow_read_to_finish();
        let report = completed(worker.stop(Duration::from_secs(1)));
        assert_eq!(report.end, CaptureWorkerEnd::Stopped);
        assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
        assert_eq!(shared.lock().release_count, 1);
        assert_eq!(mailbox.stats().accepted_frames, 0);
        assert_eq!(mailbox.stats().rejected_frames, 1);
        assert_eq!(mailbox.stats().cancelled_frames, 0);
        assert_eq!(mailbox.receive(), ReceiveOutcome::Cancelled);
        assert_eq!(gate.status(), CaptureOwnerStatus::Available);
    }

    #[test]
    fn completion_waits_for_source_release_to_return() {
        let plan = FakePlan {
            block_release: true,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let mut worker = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        shared.wait_for(Duration::from_secs(1), |state| state.release_entered);

        assert_eq!(
            worker.stop(Duration::from_millis(5)),
            CaptureWorkerWait::TimedOut
        );
        assert!(shared.lock().interrupt_called);
        assert!(matches!(gate.status(), CaptureOwnerStatus::Owned { .. }));
        shared.allow_release_to_finish();
        let report = completed(worker.wait_timeout(Duration::from_secs(1)));
        assert_eq!(report.end, CaptureWorkerEnd::Stopped);
        assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
        assert_eq!(shared.lock().release_count, 1);
        assert_eq!(gate.status(), CaptureOwnerStatus::Available);
        assert_owner_thread_for_all_source_calls(&shared);
    }

    #[test]
    fn natural_end_drains_final_frame_and_restart_has_a_fresh_session() {
        let first_plan = FakePlan {
            frame_count: 1,
            ..FakePlan::default()
        };
        let first_shared = FakeShared::new(first_plan);
        let gate = CaptureOwnerGate::new();
        let mut first = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&first_shared)),
        )
        .expect("first worker starts");
        let first_session = first.session_id().to_owned();
        let first_mailbox = first.mailbox();
        let first_report = completed(first.wait_timeout(Duration::from_secs(1)));
        assert_eq!(first_report.end, CaptureWorkerEnd::NaturalEnd);
        assert_eq!(first_report.frames_published, 1);
        assert_eq!(
            first_report.owner_disposition,
            CaptureOwnerDisposition::Released
        );
        assert!(matches!(
            first_mailbox.receive(),
            ReceiveOutcome::Frame(frame)
                if frame.source().session_id == first_session
                    && frame.source().frame_id == "frame-1"
        ));
        assert_eq!(first_mailbox.receive(), ReceiveOutcome::Closed);

        let second_plan = FakePlan {
            frame_count: 1,
            ..FakePlan::default()
        };
        let second_shared = FakeShared::new(second_plan);
        let mut second = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&second_shared)),
        )
        .expect("replacement starts after join and confirmed release");
        let second_session = second.session_id().to_owned();
        let second_mailbox = second.mailbox();
        let second_report = completed(second.wait_timeout(Duration::from_secs(1)));
        assert_eq!(second_report.end, CaptureWorkerEnd::NaturalEnd);
        assert_ne!(first_session, second_session);
        assert_eq!(second_report.session_id, second_session);
        assert!(matches!(
            second_mailbox.receive(),
            ReceiveOutcome::Frame(frame)
                if frame.source().session_id == second_session
                    && frame.source().frame_id == "frame-1"
        ));
        assert_eq!(gate.status(), CaptureOwnerStatus::Available);
    }

    #[test]
    fn read_failure_and_read_panic_are_reported_but_confirmed_release_clears_gate() {
        for (read_error, read_panics) in
            [(Some("device read failed".to_owned()), false), (None, true)]
        {
            let plan = FakePlan {
                read_error,
                read_panics,
                ..FakePlan::default()
            };
            let shared = FakeShared::new(plan);
            let gate = CaptureOwnerGate::new();
            let mut worker = CaptureWorker::start_for_test(
                gate.clone(),
                TEST_FRAME_LIMIT,
                FakeFactory(Arc::clone(&shared)),
            )
            .expect("worker starts");
            let report = completed(worker.wait_timeout(Duration::from_secs(1)));
            assert_eq!(report.end, CaptureWorkerEnd::Failed);
            assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
            assert!(if read_panics {
                report
                    .failures
                    .iter()
                    .any(|failure| matches!(failure, CaptureWorkerFailure::ReadPanicked(_)))
            } else {
                report.failures.iter().any(|failure| {
                    matches!(
                        failure,
                        CaptureWorkerFailure::Read(message) if message == "device read failed"
                    )
                })
            });
            assert_eq!(shared.lock().release_count, 1);
            assert_eq!(gate.status(), CaptureOwnerStatus::Available);
        }
    }

    #[test]
    fn release_failure_or_panic_quarantines_gate_and_blocks_future_open() {
        for (release_error, release_panics) in [
            (Some("device release failed".to_owned()), false),
            (None, true),
        ] {
            let plan = FakePlan {
                release_error,
                release_panics,
                ..FakePlan::default()
            };
            let shared = FakeShared::new(plan);
            let gate = CaptureOwnerGate::new();
            let mut worker = CaptureWorker::start_for_test(
                gate.clone(),
                TEST_FRAME_LIMIT,
                FakeFactory(Arc::clone(&shared)),
            )
            .expect("worker starts");
            let report = completed(worker.wait_timeout(Duration::from_secs(1)));
            assert_eq!(report.end, CaptureWorkerEnd::Failed);
            assert!(matches!(
                report.owner_disposition,
                CaptureOwnerDisposition::Quarantined { .. }
            ));
            assert!(if release_panics {
                report
                    .failures
                    .iter()
                    .any(|failure| matches!(failure, CaptureWorkerFailure::ReleasePanicked(_)))
            } else {
                report.failures.iter().any(|failure| {
                    matches!(
                        failure,
                        CaptureWorkerFailure::Release(message) if message == "device release failed"
                    )
                })
            });
            assert!(matches!(
                gate.status(),
                CaptureOwnerStatus::Quarantined { .. }
            ));
            let retry_shared = FakeShared::new(FakePlan::default());
            assert!(matches!(
                CaptureWorker::start_for_test(
                    gate.clone(),
                    TEST_FRAME_LIMIT,
                    FakeFactory(Arc::clone(&retry_shared))
                ),
                Err(CaptureStartError::OwnerGate(
                    CaptureOwnerGateError::Quarantined { .. }
                ))
            ));
            assert_eq!(retry_shared.lock().open_count, 0);
        }
    }

    #[test]
    fn source_destructor_panic_quarantines_after_successful_release() {
        let plan = FakePlan {
            source_drop_panics: true,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let mut worker = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        let report = completed(worker.wait_timeout(Duration::from_secs(1)));
        assert!(report
            .failures
            .iter()
            .any(|failure| matches!(failure, CaptureWorkerFailure::SourceDropPanicked(_))));
        assert!(matches!(
            report.owner_disposition,
            CaptureOwnerDisposition::Quarantined { .. }
        ));
        assert_eq!(shared.lock().release_count, 1);
        assert!(matches!(
            gate.status(),
            CaptureOwnerStatus::Quarantined { .. }
        ));
    }

    #[test]
    fn failed_open_clears_gate_after_factory_cleanup_but_open_panic_quarantines() {
        let plan = FakePlan {
            open_error: Some("camera unavailable".to_owned()),
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let mut worker = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        let report = completed(worker.wait_timeout(Duration::from_secs(1)));
        assert!(report.failures.iter().any(|failure| matches!(
            failure,
            CaptureWorkerFailure::Open(message) if message == "camera unavailable"
        )));
        assert_eq!(report.owner_disposition, CaptureOwnerDisposition::Released);
        assert_eq!(gate.status(), CaptureOwnerStatus::Available);

        let panic_plan = FakePlan {
            open_panics: true,
            ..FakePlan::default()
        };
        let panic_shared = FakeShared::new(panic_plan);
        let mut panicking = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&panic_shared)),
        )
        .expect("worker starts before factory panic");
        let panic_report = completed(panicking.wait_timeout(Duration::from_secs(1)));
        assert!(panic_report
            .failures
            .iter()
            .any(|failure| matches!(failure, CaptureWorkerFailure::OpenPanicked(_))));
        assert!(matches!(
            panic_report.owner_disposition,
            CaptureOwnerDisposition::Quarantined { .. }
        ));
        assert!(matches!(
            gate.status(),
            CaptureOwnerStatus::Quarantined { .. }
        ));
    }

    #[test]
    fn dropping_worker_interrupts_and_joins_instead_of_detaching() {
        let plan = FakePlan {
            block_first_read: true,
            ..FakePlan::default()
        };
        let shared = FakeShared::new(plan);
        let gate = CaptureOwnerGate::new();
        let worker = CaptureWorker::start_for_test(
            gate.clone(),
            TEST_FRAME_LIMIT,
            FakeFactory(Arc::clone(&shared)),
        )
        .expect("worker starts");
        shared.wait_for(Duration::from_secs(1), |state| state.read_entered);

        drop(worker);

        assert_eq!(shared.lock().release_count, 1);
        assert_eq!(gate.status(), CaptureOwnerStatus::Available);
        assert_owner_thread_for_all_source_calls(&shared);
    }
}
