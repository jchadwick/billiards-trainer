//! Opt-in, bounded timing contexts for one live frame and its nested work.
//!
//! Construct [`FrameTimingContext`] only when timing collection is enabled. Its
//! default value is disabled and holds no shared state or span buffer. Active
//! contexts record at most [`MAX_FRAME_SPANS`] spans. Span guards may move
//! between threads; dropping a guard records its monotonic completion time,
//! including during error propagation and panic unwinding.
//!
//! A frame snapshot reports the whole frame duration separately from each
//! span's inclusive and exclusive duration. Exclusive duration subtracts the
//! union of direct-child intervals after clipping them to the parent interval,
//! so overlapping children are never counted twice.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Serialize;

/// Maximum number of span records retained by one active frame context.
///
/// Attempts to add more spans are rejected and counted in the eventual
/// snapshot. This bounds storage even when instrumentation is accidentally
/// applied to a frame with a highly repetitive call path.
pub const MAX_FRAME_SPANS: usize = 256;

/// Stable identifier for a span within its frame context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct SpanId(u64);

impl SpanId {
    /// Return the numeric identifier. IDs begin at one and increase in the
    /// order spans are accepted by the context.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// An opt-in collector for timing one frame and nested spans of its work.
///
/// `Default` constructs a disabled context without allocating a collector,
/// mutex, or span buffer. Call [`FrameTimingContext::new`] to enable recording
/// for one frame. Clones refer to the same bounded frame context and are safe
/// to use from worker threads.
#[derive(Clone, Default)]
pub struct FrameTimingContext {
    shared: Option<Arc<SharedFrame>>,
}

impl std::fmt::Debug for FrameTimingContext {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FrameTimingContext")
            .field("enabled", &self.shared.is_some())
            .finish()
    }
}

impl FrameTimingContext {
    /// Start an active timing context using the supplied monotonic frame start.
    ///
    /// All timestamps passed to this context must come from the same monotonic
    /// clock domain as `frame_started_at`. The supplied instant is retained as
    /// the origin for frame duration and span offsets.
    pub fn new(frame_started_at: Instant) -> Self {
        Self {
            shared: Some(Arc::new(SharedFrame {
                frame_started_at,
                state: Mutex::new(FrameState::default()),
            })),
        }
    }

    /// Whether this context records spans. The default context is disabled.
    pub fn is_enabled(&self) -> bool {
        self.shared.is_some()
    }

    /// Begin a top-level span at the current monotonic time.
    pub fn start_span(&self, name: &str) -> Result<SpanGuard, SpanStartError> {
        self.start_span_at(name, None, Instant::now())
    }

    /// Begin a child span at the current monotonic time.
    ///
    /// The parent must exist in this context and still be open. This prevents
    /// a late asynchronous task from silently attaching work to an already
    /// completed stage.
    pub fn start_child_span(
        &self,
        name: &str,
        parent: SpanId,
    ) -> Result<SpanGuard, SpanStartError> {
        self.start_span_at(name, Some(parent), Instant::now())
    }

    /// Begin a span at an explicit instant from the context's monotonic clock.
    ///
    /// This variant also makes timing tests deterministic and lets callers
    /// propagate a timestamp captured at a well-defined pipeline boundary.
    pub fn start_span_at(
        &self,
        name: &str,
        parent: Option<SpanId>,
        started_at: Instant,
    ) -> Result<SpanGuard, SpanStartError> {
        let shared = self.shared.as_ref().ok_or(SpanStartError::Disabled)?;
        validate_name(name)?;

        let mut state = lock_state(shared);
        if started_at
            .checked_duration_since(shared.frame_started_at)
            .is_none()
        {
            return Err(SpanStartError::ClockBeforeFrameStart);
        }
        if state.spans.iter().any(|span| span.name == name) {
            return Err(SpanStartError::DuplicateName);
        }
        if let Some(parent_id) = parent {
            let parent_span = state
                .spans
                .iter()
                .find(|span| span.id == parent_id)
                .ok_or(SpanStartError::ParentNotFound)?;
            if parent_span.finished_at.is_some() {
                return Err(SpanStartError::ParentNotOpen);
            }
            if started_at
                .checked_duration_since(parent_span.started_at)
                .is_none()
            {
                return Err(SpanStartError::ClockBeforeParentStart);
            }
        }
        if state.spans.len() >= MAX_FRAME_SPANS {
            state.dropped_spans = state.dropped_spans.saturating_add(1);
            return Err(SpanStartError::CapacityReached);
        }

        // A bounded Vec grows only as actual spans are recorded. Disabled
        // contexts never reach this point or allocate any timing state.
        let id = SpanId(state.next_id);
        state.next_id = state.next_id.saturating_add(1);
        state.spans.push(SpanRecord {
            id,
            parent_id: parent,
            name: name.to_owned(),
            started_at,
            finished_at: None,
            invalid_clock: false,
        });

        Ok(SpanGuard {
            shared: Arc::clone(shared),
            id,
            completed: false,
        })
    }

    /// Capture a completed snapshot at the current monotonic time.
    pub fn snapshot(&self) -> Result<FrameTimingSnapshot, SnapshotError> {
        self.snapshot_at(Instant::now())
    }

    /// Capture a completed snapshot at an explicit monotonic instant.
    ///
    /// A snapshot is rejected while any accepted span remains open, rather
    /// than silently treating an unfinished span as a zero-duration sample.
    /// Invalid completed span clocks are preserved as records with unavailable
    /// offsets and durations and are counted in `invalid_clock_spans`.
    pub fn snapshot_at(&self, finished_at: Instant) -> Result<FrameTimingSnapshot, SnapshotError> {
        let shared = self.shared.as_ref().ok_or(SnapshotError::Disabled)?;
        let frame_elapsed = finished_at
            .checked_duration_since(shared.frame_started_at)
            .ok_or(SnapshotError::ClockBeforeFrameStart)?;
        let state = lock_state(shared);
        let unfinished_spans = state
            .spans
            .iter()
            .filter(|span| span.finished_at.is_none())
            .count();
        if unfinished_spans != 0 {
            return Err(SnapshotError::UnfinishedSpans { unfinished_spans });
        }

        let spans = state
            .spans
            .iter()
            .map(|span| snapshot_span(span, &state.spans, shared.frame_started_at))
            .collect();
        let invalid_clock_spans = state.spans.iter().filter(|span| span.invalid_clock).count();
        Ok(FrameTimingSnapshot {
            frame_elapsed_ns: duration_ns(frame_elapsed),
            spans,
            dropped_spans: state.dropped_spans,
            invalid_clock_spans,
        })
    }
}

/// RAII completion handle for one recorded span.
///
/// Dropping an unfinished guard records `Instant::now()`. `finish_at` is
/// available for deterministic timing boundaries and tests.
pub struct SpanGuard {
    shared: Arc<SharedFrame>,
    id: SpanId,
    completed: bool,
}

impl std::fmt::Debug for SpanGuard {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpanGuard")
            .field("id", &self.id)
            .field("completed", &self.completed)
            .finish()
    }
}

impl SpanGuard {
    /// Return this span's stable frame-local identifier.
    pub const fn id(&self) -> SpanId {
        self.id
    }

    /// Complete the span at the current monotonic time.
    pub fn finish(self) -> Result<(), SpanFinishError> {
        self.finish_at(Instant::now())
    }

    /// Complete the span at an explicit instant from the same monotonic clock.
    ///
    /// If the supplied instant precedes the span start, the span is closed as
    /// invalid and the error is returned. Keeping the invalid record makes the
    /// timing defect observable while ensuring it cannot produce a negative or
    /// misleading duration.
    pub fn finish_at(mut self, finished_at: Instant) -> Result<(), SpanFinishError> {
        let result = complete_span(&self.shared, self.id, finished_at);
        self.completed = true;
        result
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.completed {
            // Instant::now is from Rust's process monotonic clock. Recovering a
            // poisoned lock keeps Drop non-panicking during unwinding.
            let _ = complete_span(&self.shared, self.id, Instant::now());
            self.completed = true;
        }
    }
}

/// Completed, bounded timing data for one frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrameTimingSnapshot {
    /// Elapsed time from the supplied frame start to snapshot capture.
    pub frame_elapsed_ns: u64,
    /// Span records in stable ID/acceptance order.
    pub spans: Vec<SpanTiming>,
    /// Span creation attempts rejected because the context was full.
    pub dropped_spans: u64,
    /// Accepted spans closed with an invalid backwards clock reading.
    pub invalid_clock_spans: usize,
}

/// Exportable timing data for one accepted span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpanTiming {
    pub id: SpanId,
    pub parent_id: Option<SpanId>,
    pub name: String,
    /// Span start offset from frame start. Unavailable for invalid clocks.
    pub offset_ns: Option<u64>,
    /// Span's actual start-to-finish duration, including child work.
    pub inclusive_elapsed_ns: Option<u64>,
    /// Inclusive time less the union of direct-child intervals clipped to the
    /// span. Unavailable when this span's own clock is invalid.
    pub exclusive_elapsed_ns: Option<u64>,
    /// Whether the span used a valid non-negative interval.
    pub clock_valid: bool,
}

/// Errors encountered while beginning a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanStartError {
    Disabled,
    InvalidName,
    DuplicateName,
    ParentNotFound,
    ParentNotOpen,
    ClockBeforeFrameStart,
    ClockBeforeParentStart,
    CapacityReached,
}

impl Display for SpanStartError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Disabled => "frame timing is disabled",
            Self::InvalidName => "span name is empty, too long, or contains control characters",
            Self::DuplicateName => "span name already exists in this frame",
            Self::ParentNotFound => "parent span does not exist in this frame",
            Self::ParentNotOpen => "parent span has already finished",
            Self::ClockBeforeFrameStart => "span clock precedes frame start",
            Self::ClockBeforeParentStart => "span clock precedes parent start",
            Self::CapacityReached => "frame span capacity reached",
        };
        formatter.write_str(message)
    }
}

impl Error for SpanStartError {}

/// Error returned when explicitly completing a span with a backwards clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanFinishError {
    ClockBeforeSpanStart,
    SpanNotFound,
    SpanAlreadyFinished,
}

impl Display for SpanFinishError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::ClockBeforeSpanStart => "span finish clock precedes span start",
            Self::SpanNotFound => "span does not exist in this frame",
            Self::SpanAlreadyFinished => "span has already finished",
        };
        formatter.write_str(message)
    }
}

impl Error for SpanFinishError {}

/// Errors encountered while producing a frame snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotError {
    Disabled,
    ClockBeforeFrameStart,
    UnfinishedSpans { unfinished_spans: usize },
}

impl Display for SnapshotError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => formatter.write_str("frame timing is disabled"),
            Self::ClockBeforeFrameStart => {
                formatter.write_str("snapshot clock precedes frame start")
            }
            Self::UnfinishedSpans { unfinished_spans } => {
                write!(
                    formatter,
                    "{unfinished_spans} frame timing spans are unfinished"
                )
            }
        }
    }
}

impl Error for SnapshotError {}

struct SharedFrame {
    frame_started_at: Instant,
    state: Mutex<FrameState>,
}

struct FrameState {
    next_id: u64,
    spans: Vec<SpanRecord>,
    dropped_spans: u64,
}

impl Default for FrameState {
    fn default() -> Self {
        Self {
            next_id: 1,
            spans: Vec::new(),
            dropped_spans: 0,
        }
    }
}

struct SpanRecord {
    id: SpanId,
    parent_id: Option<SpanId>,
    name: String,
    started_at: Instant,
    finished_at: Option<Instant>,
    invalid_clock: bool,
}

fn validate_name(name: &str) -> Result<(), SpanStartError> {
    const MAX_SPAN_NAME_BYTES: usize = 128;
    if name.trim().is_empty()
        || name.len() > MAX_SPAN_NAME_BYTES
        || name.chars().any(char::is_control)
    {
        return Err(SpanStartError::InvalidName);
    }
    Ok(())
}

fn lock_state(shared: &SharedFrame) -> MutexGuard<'_, FrameState> {
    shared
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn complete_span(
    shared: &SharedFrame,
    id: SpanId,
    finished_at: Instant,
) -> Result<(), SpanFinishError> {
    let mut state = lock_state(shared);
    let span = state
        .spans
        .iter_mut()
        .find(|span| span.id == id)
        .ok_or(SpanFinishError::SpanNotFound)?;
    if span.finished_at.is_some() {
        return Err(SpanFinishError::SpanAlreadyFinished);
    }
    span.invalid_clock = finished_at
        .checked_duration_since(span.started_at)
        .is_none();
    span.finished_at = Some(finished_at);
    if span.invalid_clock {
        Err(SpanFinishError::ClockBeforeSpanStart)
    } else {
        Ok(())
    }
}

fn snapshot_span(span: &SpanRecord, all_spans: &[SpanRecord], frame_start: Instant) -> SpanTiming {
    if span.invalid_clock {
        return SpanTiming {
            id: span.id,
            parent_id: span.parent_id,
            name: span.name.clone(),
            offset_ns: None,
            inclusive_elapsed_ns: None,
            exclusive_elapsed_ns: None,
            clock_valid: false,
        };
    }

    let offset_ns = span
        .started_at
        .checked_duration_since(frame_start)
        .map(duration_ns);
    let inclusive = span
        .finished_at
        .and_then(|finished_at| finished_at.checked_duration_since(span.started_at));
    let inclusive_elapsed_ns = inclusive.map(duration_ns);
    let exclusive_elapsed_ns = inclusive.map(|inclusive| {
        duration_ns(inclusive).saturating_sub(direct_children_union_ns(span, all_spans))
    });

    SpanTiming {
        id: span.id,
        parent_id: span.parent_id,
        name: span.name.clone(),
        offset_ns,
        inclusive_elapsed_ns,
        exclusive_elapsed_ns,
        clock_valid: offset_ns.is_some() && inclusive_elapsed_ns.is_some(),
    }
}

fn direct_children_union_ns(parent: &SpanRecord, all_spans: &[SpanRecord]) -> u64 {
    let Some(parent_finished_at) = parent.finished_at else {
        return 0;
    };
    let mut intervals = all_spans
        .iter()
        .filter(|child| child.parent_id == Some(parent.id) && !child.invalid_clock)
        .filter_map(|child| {
            let child_finished_at = child.finished_at?;
            let start = child.started_at.max(parent.started_at);
            let end = child_finished_at.min(parent_finished_at);
            (end > start).then_some((start, end))
        })
        .collect::<Vec<_>>();
    intervals.sort_unstable_by_key(|(start, _)| *start);

    let mut union_ns = 0_u64;
    let mut current: Option<(Instant, Instant)> = None;
    for (start, end) in intervals {
        match current {
            Some((current_start, current_end)) if start <= current_end => {
                current = Some((current_start, current_end.max(end)));
            }
            Some((current_start, current_end)) => {
                union_ns = union_ns.saturating_add(duration_ns(
                    current_end
                        .checked_duration_since(current_start)
                        .unwrap_or(Duration::ZERO),
                ));
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        union_ns = union_ns.saturating_add(duration_ns(
            end.checked_duration_since(start).unwrap_or(Duration::ZERO),
        ));
    }
    union_ns
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::thread;

    fn at(base: Instant, nanoseconds: u64) -> Instant {
        base.checked_add(Duration::from_nanos(nanoseconds))
            .expect("small test instant should be representable")
    }

    #[test]
    fn nested_spans_report_actual_frame_inclusive_and_exclusive_durations() {
        let base = Instant::now();
        let context = FrameTimingContext::new(base);
        let parent = context
            .start_span_at("inference", None, at(base, 10))
            .unwrap();
        let parent_id = parent.id();
        let child = context
            .start_span_at("preprocess", Some(parent_id), at(base, 20))
            .unwrap();
        child.finish_at(at(base, 30)).unwrap();
        parent.finish_at(at(base, 50)).unwrap();

        let snapshot = context.snapshot_at(at(base, 80)).unwrap();
        assert_eq!(snapshot.frame_elapsed_ns, 80);
        assert_eq!(snapshot.spans.len(), 2);
        assert_eq!(snapshot.spans[0].id.get(), 1);
        assert_eq!(snapshot.spans[0].offset_ns, Some(10));
        assert_eq!(snapshot.spans[0].inclusive_elapsed_ns, Some(40));
        assert_eq!(snapshot.spans[0].exclusive_elapsed_ns, Some(30));
        assert_eq!(snapshot.spans[1].parent_id, Some(parent_id));
        assert_eq!(snapshot.spans[1].inclusive_elapsed_ns, Some(10));
        assert_eq!(snapshot.spans[1].exclusive_elapsed_ns, Some(10));
    }

    #[test]
    fn overlapping_siblings_subtract_their_interval_union_only_once() {
        let base = Instant::now();
        let context = FrameTimingContext::new(base);
        let parent = context
            .start_span_at("frame-work", None, at(base, 10))
            .unwrap();
        let parent_id = parent.id();
        let first = context
            .start_span_at("ball-inference", Some(parent_id), at(base, 20))
            .unwrap();
        let second = context
            .start_span_at("cue-inference", Some(parent_id), at(base, 30))
            .unwrap();
        first.finish_at(at(base, 50)).unwrap();
        second.finish_at(at(base, 55)).unwrap();
        parent.finish_at(at(base, 60)).unwrap();

        let snapshot = context.snapshot_at(at(base, 70)).unwrap();
        assert_eq!(snapshot.spans[0].inclusive_elapsed_ns, Some(50));
        // Child intervals [20, 50] and [30, 55] form one 35 ns union.
        assert_eq!(snapshot.spans[0].exclusive_elapsed_ns, Some(15));
    }

    #[test]
    fn guards_complete_when_scoped_work_returns_an_error_or_unwinds() {
        let base = Instant::now();
        let context = FrameTimingContext::new(base);
        let context_for_error = context.clone();
        let error: Result<(), &'static str> = {
            let _span = context_for_error.start_span("fallible-stage").unwrap();
            Err("stage failed")
        };
        assert_eq!(error, Err("stage failed"));

        let context_for_panic = context.clone();
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _span = context_for_panic.start_span("panicking-stage").unwrap();
            panic!("expected test unwind");
        }));
        assert!(panic.is_err());

        let snapshot = context.snapshot().unwrap();
        assert_eq!(snapshot.spans.len(), 2);
        assert!(snapshot
            .spans
            .iter()
            .all(|span| span.clock_valid && span.inclusive_elapsed_ns.is_some()));
    }

    #[test]
    fn a_span_guard_can_finish_on_another_thread() {
        let base = Instant::now();
        let context = FrameTimingContext::new(base);
        let guard = context.start_span_at("worker", None, at(base, 5)).unwrap();
        let worker = thread::spawn(move || guard.finish_at(at(base, 15)));
        worker.join().unwrap().unwrap();

        let snapshot = context.snapshot_at(at(base, 20)).unwrap();
        assert_eq!(snapshot.spans[0].inclusive_elapsed_ns, Some(10));
    }

    #[test]
    fn duplicate_invalid_and_unfinished_states_are_rejected() {
        let base = Instant::now();
        let context = FrameTimingContext::new(base);
        let first = context.start_span_at("inference", None, base).unwrap();
        assert_eq!(
            context.start_span_at("inference", None, base).unwrap_err(),
            SpanStartError::DuplicateName
        );
        assert_eq!(
            context.start_span_at(" \n ", None, base).unwrap_err(),
            SpanStartError::InvalidName
        );
        assert_eq!(
            context.snapshot_at(base).unwrap_err(),
            SnapshotError::UnfinishedSpans {
                unfinished_spans: 1
            }
        );
        first.finish_at(base).unwrap();

        let parent = context.start_span_at("parent", None, base).unwrap();
        let parent_id = parent.id();
        parent.finish_at(base).unwrap();
        assert_eq!(
            context
                .start_span_at("late-child", Some(parent_id), base)
                .unwrap_err(),
            SpanStartError::ParentNotOpen
        );
        assert_eq!(
            context
                .start_span_at("missing-child", Some(SpanId(u64::MAX)), base)
                .unwrap_err(),
            SpanStartError::ParentNotFound
        );
    }

    #[test]
    fn invalid_backwards_clock_is_reported_without_exporting_negative_times() {
        let base = Instant::now();
        let context = FrameTimingContext::new(base);
        let span = context.start_span_at("stage", None, at(base, 20)).unwrap();
        assert_eq!(
            span.finish_at(at(base, 10)),
            Err(SpanFinishError::ClockBeforeSpanStart)
        );

        let snapshot = context.snapshot_at(at(base, 30)).unwrap();
        assert_eq!(snapshot.invalid_clock_spans, 1);
        assert!(!snapshot.spans[0].clock_valid);
        assert_eq!(snapshot.spans[0].offset_ns, None);
        assert_eq!(snapshot.spans[0].inclusive_elapsed_ns, None);
        assert_eq!(snapshot.spans[0].exclusive_elapsed_ns, None);
        assert_eq!(
            context
                .snapshot_at(base.checked_sub(Duration::from_nanos(1)).unwrap())
                .unwrap_err(),
            SnapshotError::ClockBeforeFrameStart
        );
    }

    #[test]
    fn bounded_context_counts_capacity_drops_and_preserves_accepted_records() {
        let base = Instant::now();
        let context = FrameTimingContext::new(base);
        for index in 0..MAX_FRAME_SPANS {
            context
                .start_span_at(&format!("stage-{index}"), None, base)
                .unwrap()
                .finish_at(base)
                .unwrap();
        }
        for _ in 0..3 {
            assert_eq!(
                context.start_span_at("overflow", None, base).unwrap_err(),
                SpanStartError::CapacityReached
            );
        }

        let snapshot = context.snapshot_at(base).unwrap();
        assert_eq!(snapshot.spans.len(), MAX_FRAME_SPANS);
        assert_eq!(snapshot.dropped_spans, 3);
        assert_eq!(snapshot.spans[0].id.get(), 1);
        assert_eq!(
            snapshot.spans[MAX_FRAME_SPANS - 1].id.get() as usize,
            MAX_FRAME_SPANS
        );
    }

    #[test]
    fn default_context_is_disabled_without_accepting_or_exporting_spans() {
        let context = FrameTimingContext::default();
        assert!(!context.is_enabled());
        assert_eq!(
            context.start_span("unused").unwrap_err(),
            SpanStartError::Disabled
        );
        assert_eq!(context.snapshot().unwrap_err(), SnapshotError::Disabled);
    }
}
