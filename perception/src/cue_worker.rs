//! Single-flight worker for cue-shaft geometry refinement.
//!
//! Cue geometry is useful at a higher cadence than the ball detector, but it
//! still must not build a frame queue. The caller transfers one owned capture
//! lease only while this worker is waiting and has no unconsumed result. A
//! refused submit returns the same lease to the caller, preserving capture
//! memory accounting and avoiding an RGB pixel copy.

use std::any::Any;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::capture::OwnedCaptureFrame;
use crate::cue_geometry::{
    estimate_cue_geometry, CueGeometry, CueGeometryConfig, CueGeometryError, CueProposal,
};

/// Identity and processing revisions required to associate cue geometry with
/// the exact source observation and phase context that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueWorkTag {
    pub source_session_id: String,
    pub frame_id: String,
    pub configuration_revision: u64,
    pub calibration_revision: u64,
    pub motion_generation: u64,
}

/// Process-local timing for one cue estimate. `observation_age_at_submission`
/// is provided by the coordinator in its authoritative observation clock;
/// elapsed worker durations are measured with the same process-local monotonic
/// clock as the capture lease's read-completion time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueObservationTiming {
    pub frame_read_completed_at: Instant,
    pub submitted_at: Instant,
    pub processing_started_at: Instant,
    pub processing_completed_at: Instant,
    pub observation_age_at_submission: Duration,
    pub observation_age_at_completion: Duration,
}

/// The exact capture lease and context used for one cue geometry estimate.
#[derive(Debug)]
pub struct CueGeometryCompletion {
    pub tag: CueWorkTag,
    pub frame: OwnedCaptureFrame,
    pub timing: CueObservationTiming,
    pub result: Result<Option<CueGeometry>, CueGeometryWorkerFailure>,
}

/// A recoverable failure from cue estimation. A bad cue estimate only removes
/// cue guidance; it does not invalidate otherwise valid ball observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CueGeometryWorkerFailure {
    pub kind: CueGeometryWorkerFailureKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueGeometryWorkerFailureKind {
    InvalidConfiguration,
    Geometry,
    EstimatorPanicked,
    WorkerPanicked,
}

impl Display for CueGeometryWorkerFailureKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfiguration => "invalid cue geometry configuration",
            Self::Geometry => "cue geometry estimation failed",
            Self::EstimatorPanicked => "cue geometry estimator panicked",
            Self::WorkerPanicked => "cue geometry worker panicked",
        })
    }
}

impl Display for CueGeometryWorkerFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.kind, self.message)
    }
}

/// Why a nonblocking cue dispatch was refused. Every variant returns the
/// original frame lease so the caller can continue using or release it.
#[derive(Debug)]
pub enum CueGeometrySubmitError {
    Busy(OwnedCaptureFrame),
    ShuttingDown(OwnedCaptureFrame),
    WorkerUnavailable(OwnedCaptureFrame),
    TooManyProposals {
        frame: OwnedCaptureFrame,
        maximum: usize,
        actual: usize,
    },
    ProposalAllocationFailed(OwnedCaptureFrame),
    TagDoesNotMatchFrame {
        frame: OwnedCaptureFrame,
        expected_session_id: String,
        expected_frame_id: String,
    },
}

impl CueGeometrySubmitError {
    /// Recover the frame lease rejected by this dispatch attempt.
    pub fn into_frame(self) -> OwnedCaptureFrame {
        match self {
            Self::Busy(frame)
            | Self::ShuttingDown(frame)
            | Self::WorkerUnavailable(frame)
            | Self::ProposalAllocationFailed(frame)
            | Self::TooManyProposals { frame, .. }
            | Self::TagDoesNotMatchFrame { frame, .. } => frame,
        }
    }

    pub fn reason(&self) -> CueGeometrySubmitRefusal {
        match self {
            Self::Busy(_) => CueGeometrySubmitRefusal::Busy,
            Self::ShuttingDown(_) => CueGeometrySubmitRefusal::ShuttingDown,
            Self::WorkerUnavailable(_) => CueGeometrySubmitRefusal::WorkerUnavailable,
            Self::TooManyProposals { .. } => CueGeometrySubmitRefusal::TooManyProposals,
            Self::ProposalAllocationFailed(_) => CueGeometrySubmitRefusal::ProposalAllocationFailed,
            Self::TagDoesNotMatchFrame { .. } => CueGeometrySubmitRefusal::TagDoesNotMatchFrame,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueGeometrySubmitRefusal {
    Busy,
    ShuttingDown,
    WorkerUnavailable,
    TooManyProposals,
    ProposalAllocationFailed,
    TagDoesNotMatchFrame,
}

/// Result of requesting a bounded cue worker shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueGeometryWorkerShutdown {
    Stopped,
    /// The estimator did not stop within the supplied bound. The thread is
    /// detached; a supervisor can contain a stuck worker by stopping the
    /// perception process.
    TimedOutDetached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueGeometryWorkerState {
    Ready,
    WorkerWaking,
    EstimationInFlight,
    ResultReady,
    ShuttingDown,
    Stopped,
    Failed,
}

/// Startup failure before the worker accepts any frame.
#[derive(Debug)]
pub enum CueGeometryWorkerStartError {
    InvalidConfiguration(CueGeometryError),
    ThreadSpawn(std::io::Error),
}

impl Display for CueGeometryWorkerStartError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(error) => {
                write!(
                    formatter,
                    "cue geometry configuration is invalid: {error:?}"
                )
            }
            Self::ThreadSpawn(error) => write!(formatter, "could not spawn cue worker: {error}"),
        }
    }
}

impl std::error::Error for CueGeometryWorkerStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidConfiguration(_) => None,
            Self::ThreadSpawn(error) => Some(error),
        }
    }
}

struct CueGeometryJob {
    tag: CueWorkTag,
    frame: OwnedCaptureFrame,
    proposals: Vec<CueProposal>,
    submitted_at: Instant,
    observation_age_at_submission: Duration,
}

struct WorkerState {
    job: Option<CueGeometryJob>,
    completion: Option<CueGeometryCompletion>,
    outstanding: bool,
    worker_waiting: bool,
    shutting_down: bool,
    worker_exited: bool,
    terminal_failure: Option<CueGeometryWorkerFailure>,
}

struct Shared {
    state: Mutex<WorkerState>,
    changed: Condvar,
}

type Estimator = dyn Fn(
        &[u8],
        u32,
        u32,
        &[CueProposal],
        Duration,
        CueGeometryConfig,
    ) -> Result<Option<CueGeometry>, CueGeometryError>
    + Send
    + 'static;

/// Handle for one cue-geometry thread. Dropping it requests shutdown and
/// detaches instead of blocking on an estimator that fails to return.
pub struct CueGeometryWorker {
    shared: Arc<Shared>,
    join: Option<JoinHandle<()>>,
    maximum_proposals: usize,
    _not_generic: PhantomData<fn()>,
}

impl std::fmt::Debug for CueGeometryWorker {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CueGeometryWorker")
            .field("state", &self.state())
            .field("maximum_proposals", &self.maximum_proposals)
            .field(
                "thread_finished",
                &self.join.as_ref().is_none_or(JoinHandle::is_finished),
            )
            .finish()
    }
}

impl CueGeometryWorker {
    /// Start the dedicated cue worker with a validated, bounded estimator
    /// configuration. The worker is ready for its first direct handoff before
    /// this function returns.
    pub fn spawn(config: CueGeometryConfig) -> Result<Self, CueGeometryWorkerStartError> {
        Self::spawn_inner(config, |pixels, width, height, proposals, age, config| {
            estimate_cue_geometry(pixels, width, height, proposals, age, config)
        })
    }

    fn spawn_inner<F>(
        config: CueGeometryConfig,
        estimator: F,
    ) -> Result<Self, CueGeometryWorkerStartError>
    where
        F: Fn(
                &[u8],
                u32,
                u32,
                &[CueProposal],
                Duration,
                CueGeometryConfig,
            ) -> Result<Option<CueGeometry>, CueGeometryError>
            + Send
            + 'static,
    {
        let config = config
            .validate()
            .map_err(CueGeometryWorkerStartError::InvalidConfiguration)?;
        let shared = Arc::new(Shared {
            state: Mutex::new(WorkerState {
                job: None,
                completion: None,
                outstanding: false,
                worker_waiting: false,
                shutting_down: false,
                worker_exited: false,
                terminal_failure: None,
            }),
            changed: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let estimator: Box<Estimator> = Box::new(estimator);
        let join = thread::Builder::new()
            .name("cue-geometry-worker".to_owned())
            .spawn(move || run_worker(config, estimator, worker_shared))
            .map_err(CueGeometryWorkerStartError::ThreadSpawn)?;

        // Avoid dropping the first cue opportunity only because the OS has
        // not scheduled the new worker yet.
        let mut state = lock_state(&shared);
        while !state.worker_waiting && !state.worker_exited {
            state = shared
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        drop(state);

        Ok(Self {
            shared,
            join: Some(join),
            maximum_proposals: config.max_proposals,
            _not_generic: PhantomData,
        })
    }

    /// Transfer one frame without waiting. There is no pending frame queue:
    /// success requires an already-waiting worker and an empty result slot.
    /// Proposal regions are copied into a bounded job so callers can reuse
    /// their detector output independently of the cue worker.
    #[allow(clippy::result_large_err)]
    pub fn try_submit(
        &mut self,
        tag: CueWorkTag,
        frame: OwnedCaptureFrame,
        proposals: &[CueProposal],
        observation_age: Duration,
    ) -> Result<(), CueGeometrySubmitError> {
        let source = frame.source();
        if source.session_id != tag.source_session_id || source.frame_id != tag.frame_id {
            return Err(CueGeometrySubmitError::TagDoesNotMatchFrame {
                frame,
                expected_session_id: tag.source_session_id,
                expected_frame_id: tag.frame_id,
            });
        }
        if proposals.len() > self.maximum_proposals {
            return Err(CueGeometrySubmitError::TooManyProposals {
                frame,
                maximum: self.maximum_proposals,
                actual: proposals.len(),
            });
        }

        let mut state = lock_state(&self.shared);
        if state.shutting_down {
            return Err(CueGeometrySubmitError::ShuttingDown(frame));
        }
        if state.worker_exited || state.terminal_failure.is_some() {
            return Err(CueGeometrySubmitError::WorkerUnavailable(frame));
        }
        if state.outstanding || !state.worker_waiting {
            return Err(CueGeometrySubmitError::Busy(frame));
        }

        let mut copied_proposals = Vec::new();
        if copied_proposals.try_reserve_exact(proposals.len()).is_err() {
            return Err(CueGeometrySubmitError::ProposalAllocationFailed(frame));
        }
        copied_proposals.extend_from_slice(proposals);
        state.outstanding = true;
        state.worker_waiting = false;
        state.job = Some(CueGeometryJob {
            tag,
            frame,
            proposals: copied_proposals,
            submitted_at: Instant::now(),
            observation_age_at_submission: observation_age,
        });
        self.shared.changed.notify_one();
        Ok(())
    }

    /// Take a completed estimate if available. Until consumed, the worker is
    /// single-flight and another submit is refused as busy.
    pub fn poll_result(&mut self) -> Option<CueGeometryCompletion> {
        let mut state = lock_state(&self.shared);
        let completion = state.completion.take()?;
        state.outstanding = false;
        self.shared.changed.notify_all();
        Some(completion)
    }

    pub fn state(&self) -> CueGeometryWorkerState {
        let state = lock_state(&self.shared);
        if state.terminal_failure.is_some() {
            CueGeometryWorkerState::Failed
        } else if state.worker_exited {
            CueGeometryWorkerState::Stopped
        } else if state.shutting_down {
            CueGeometryWorkerState::ShuttingDown
        } else if state.completion.is_some() {
            CueGeometryWorkerState::ResultReady
        } else if state.outstanding {
            CueGeometryWorkerState::EstimationInFlight
        } else if state.worker_waiting {
            CueGeometryWorkerState::Ready
        } else {
            CueGeometryWorkerState::WorkerWaking
        }
    }

    /// Whether a frame can be transferred immediately without queueing.
    pub fn can_accept_work(&self) -> bool {
        let state = lock_state(&self.shared);
        !state.shutting_down
            && !state.worker_exited
            && state.terminal_failure.is_none()
            && !state.outstanding
            && state.worker_waiting
    }

    /// Failure outside an individual estimation result, such as an unexpected
    /// panic in worker coordination.
    pub fn terminal_failure(&self) -> Option<CueGeometryWorkerFailure> {
        lock_state(&self.shared).terminal_failure.clone()
    }

    /// Request shutdown and wait no longer than `timeout`. A stuck estimator
    /// is detached so destroying the coordinator remains bounded.
    pub fn shutdown(&mut self, timeout: Duration) -> CueGeometryWorkerShutdown {
        {
            let mut state = lock_state(&self.shared);
            state.shutting_down = true;
            self.shared.changed.notify_all();
        }

        let started = Instant::now();
        let mut state = lock_state(&self.shared);
        while !state.worker_exited {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                drop(state);
                self.join.take();
                return CueGeometryWorkerShutdown::TimedOutDetached;
            }
            let (next_state, wait_result) = self
                .shared
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if wait_result.timed_out() && !state.worker_exited {
                drop(state);
                self.join.take();
                return CueGeometryWorkerShutdown::TimedOutDetached;
            }
        }
        drop(state);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        CueGeometryWorkerShutdown::Stopped
    }
}

impl Drop for CueGeometryWorker {
    fn drop(&mut self) {
        {
            let mut state = lock_state(&self.shared);
            state.shutting_down = true;
            self.shared.changed.notify_all();
        }
        // Dropping JoinHandle detaches. Never let a nonresponsive estimator
        // turn ordinary owner drop into an unbounded join.
        self.join.take();
    }
}

fn run_worker(config: CueGeometryConfig, estimator: Box<Estimator>, shared: Arc<Shared>) {
    let result = catch_unwind(AssertUnwindSafe(|| worker_loop(config, estimator, &shared)));
    let mut state = lock_state(&shared);
    if let Err(payload) = result {
        state.terminal_failure = Some(CueGeometryWorkerFailure {
            kind: CueGeometryWorkerFailureKind::WorkerPanicked,
            message: panic_payload_message(payload),
        });
    }
    state.worker_waiting = false;
    state.worker_exited = true;
    shared.changed.notify_all();
}

fn worker_loop(config: CueGeometryConfig, estimator: Box<Estimator>, shared: &Arc<Shared>) {
    loop {
        let job = {
            let mut state = lock_state(shared);
            loop {
                if let Some(job) = state.job.take() {
                    state.worker_waiting = false;
                    break Some(job);
                }
                if state.shutting_down {
                    state.worker_waiting = false;
                    shared.changed.notify_all();
                    break None;
                }
                state.worker_waiting = true;
                shared.changed.notify_all();
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.worker_waiting = false;
            }
        };

        let Some(job) = job else {
            return;
        };

        let processing_started_at = Instant::now();
        let estimator_age = job
            .observation_age_at_submission
            .saturating_add(processing_started_at.saturating_duration_since(job.submitted_at));
        let estimation = catch_unwind(AssertUnwindSafe(|| {
            estimator(
                job.frame.rgb8(),
                job.frame.source().width_px,
                job.frame.source().height_px,
                &job.proposals,
                estimator_age,
                config,
            )
        }));
        let processing_completed_at = Instant::now();
        let observation_age_at_completion = job
            .observation_age_at_submission
            .saturating_add(processing_completed_at.saturating_duration_since(job.submitted_at));
        let result = match estimation {
            Ok(Ok(mut geometry)) => {
                if let Some(geometry) = geometry.as_mut() {
                    geometry.observation_age_ms =
                        duration_millis_saturated(observation_age_at_completion);
                }
                Ok(geometry)
            }
            Ok(Err(error)) => Err(CueGeometryWorkerFailure {
                kind: CueGeometryWorkerFailureKind::Geometry,
                message: format!("{error:?}"),
            }),
            Err(payload) => Err(CueGeometryWorkerFailure {
                kind: CueGeometryWorkerFailureKind::EstimatorPanicked,
                message: panic_payload_message(payload),
            }),
        };
        let completion = CueGeometryCompletion {
            tag: job.tag,
            timing: CueObservationTiming {
                frame_read_completed_at: job.frame.host_read_completed_at(),
                submitted_at: job.submitted_at,
                processing_started_at,
                processing_completed_at,
                observation_age_at_submission: job.observation_age_at_submission,
                observation_age_at_completion,
            },
            frame: job.frame,
            result,
        };

        let mut state = lock_state(shared);
        // A second result cannot be produced until the caller consumes this
        // completion and submits another frame.
        debug_assert!(state.completion.is_none());
        state.completion = Some(completion);
        shared.changed.notify_all();
    }
}

fn lock_state(shared: &Shared) -> MutexGuard<'_, WorkerState> {
    shared
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn duration_millis_saturated(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CueGeometrySubmitError, CueGeometrySubmitRefusal, CueGeometryWorker,
        CueGeometryWorkerFailureKind, CueGeometryWorkerShutdown, CueGeometryWorkerState,
        CueWorkTag,
    };
    use crate::capture::{
        CaptureAggregateFrameLimit, CaptureAppliedCameraControls, CaptureAppliedFrameGeometry,
        CaptureCapability, CaptureControlReadback, CaptureFrameBudget, CaptureFrameLimit,
        CaptureFrameRate, CaptureInterrupt, CaptureOpenError, CapturePixelFormat,
        CaptureReadOutcome, CaptureReceiveOutcome, CaptureRuntimeCoordinator, CaptureSource,
        CaptureSourceCapabilities, CaptureSourceFactory, CaptureSourceMetadata, CaptureStartError,
        CaptureTimestampSemantics, CaptureVideoProfile, OwnedCaptureFrame, RgbFrameEnvelope,
    };
    use crate::cue_geometry::{CueGeometryConfig, CueProposal};
    use crate::detection_batch::SourceFrame;
    use crate::ImageSpace;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    const WIDTH: u32 = 220;
    const HEIGHT: u32 = 140;
    const FRAME_BYTES: usize = (WIDTH * HEIGHT * 3) as usize;
    static CAPTURE_RUNTIME_LOCK: Mutex<()> = Mutex::new(());

    struct TestInterrupt {
        interrupted: Arc<AtomicBool>,
        wake: mpsc::Sender<()>,
    }

    impl CaptureInterrupt for TestInterrupt {
        fn interrupt(&self) {
            self.interrupted.store(true, Ordering::Release);
            let _ = self.wake.send(());
        }
    }

    struct TestFactory {
        metadata: CaptureSourceMetadata,
        interrupted: Arc<AtomicBool>,
        wake: mpsc::Sender<()>,
        requests: mpsc::Receiver<()>,
    }

    impl CaptureSourceFactory for TestFactory {
        fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
            Arc::new(TestInterrupt {
                interrupted: Arc::clone(&self.interrupted),
                wake: self.wake.clone(),
            })
        }

        fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
            Ok(Box::new(TestSource {
                metadata: self.metadata,
                interrupted: self.interrupted,
                wake: self.wake,
                requests: self.requests,
                frame_number: 0,
            }))
        }
    }

    struct TestSource {
        metadata: CaptureSourceMetadata,
        interrupted: Arc<AtomicBool>,
        wake: mpsc::Sender<()>,
        requests: mpsc::Receiver<()>,
        frame_number: u64,
    }

    impl CaptureSource for TestSource {
        fn metadata(&self) -> CaptureSourceMetadata {
            self.metadata.clone()
        }

        fn read(
            &mut self,
            session_id: &str,
            frame_limit: CaptureFrameLimit,
            frame_budget: &CaptureFrameBudget,
        ) -> Result<CaptureReadOutcome, String> {
            loop {
                if self.interrupted.load(Ordering::Acquire) {
                    return Ok(CaptureReadOutcome::End);
                }
                match self.requests.recv_timeout(Duration::from_millis(20)) {
                    Ok(()) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Ok(CaptureReadOutcome::End)
                    }
                }
            }

            self.frame_number += 1;
            let mut pixels = frame_budget
                .allocate_rgb8_buffer(WIDTH, HEIGHT, frame_limit)
                .map_err(|error| error.to_string())?;
            for pixel in pixels.as_mut_slice().as_chunks_mut::<3>().0 {
                pixel.copy_from_slice(&[30, 100, 30]);
            }
            // Include a clear shaft so the normal estimator path can be
            // exercised without relying on model output or a prerecorded file.
            draw_line(pixels.as_mut_slice(), [30.0, 108.0], [166.0, 35.0]);
            let frame = RgbFrameEnvelope::new(
                SourceFrame {
                    session_id: session_id.to_owned(),
                    frame_id: format!("frame-{}", self.frame_number),
                    timestamp: None,
                    width_px: WIDTH,
                    height_px: HEIGHT,
                },
                pixels,
                Instant::now(),
                frame_limit,
            )
            .map_err(|error| error.to_string())?;
            Ok(CaptureReadOutcome::Frame(frame))
        }

        fn release(&mut self) -> Result<(), String> {
            let _ = self.wake.send(());
            Ok(())
        }
    }

    struct CaptureHarness {
        runtime: CaptureRuntimeCoordinator,
        receiver: crate::capture::CaptureFrameReceiver,
        requests: mpsc::Sender<()>,
        _capture_serial: std::sync::MutexGuard<'static, ()>,
    }

    impl CaptureHarness {
        fn start() -> Self {
            let capture_serial = crate::capture::CAPTURE_RUNTIME_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let frame_limit = CaptureFrameLimit::new(FRAME_BYTES).unwrap();
            // Match the crate-wide test runtime budget so these tests can run
            // alongside the capture module's coordinator tests in one process.
            let aggregate_limit = CaptureAggregateFrameLimit::new(1_048_576).unwrap();
            let format = CapturePixelFormat::new("RGB8").unwrap();
            let profile = CaptureVideoProfile::new(
                WIDTH,
                HEIGHT,
                CaptureFrameRate::new(30, 1).unwrap(),
                format.clone(),
            )
            .unwrap();
            let capabilities = CaptureSourceCapabilities::new(
                Some(vec![format]),
                CaptureCapability::Unsupported,
                CaptureCapability::Unsupported,
                CaptureCapability::Unsupported,
                CaptureCapability::Unsupported,
            )
            .unwrap();
            let controls = CaptureAppliedCameraControls::new(
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
                CaptureControlReadback::Unknown,
            )
            .unwrap();
            let metadata = CaptureSourceMetadata::new(
                "cue-worker-test-camera",
                None,
                "cue-worker-test",
                profile.clone(),
                profile,
                capabilities,
                CaptureAppliedFrameGeometry::new(1, "cue-worker-test-profile", ImageSpace::Raw, 0)
                    .unwrap(),
                controls,
                CaptureTimestampSemantics::ReadCompletionOnly,
            )
            .unwrap();
            // The production coordinator uses a process-wide source gate.
            // Other unit tests may briefly hold it, so retry only the expected
            // contention result instead of making test order significant.
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let interrupted = Arc::new(AtomicBool::new(false));
                let (requests, request_receiver) = mpsc::channel();
                let (wake, _wake_receiver) = mpsc::channel();
                let factory = TestFactory {
                    metadata: metadata.clone(),
                    interrupted,
                    wake,
                    requests: request_receiver,
                };
                let mut runtime =
                    CaptureRuntimeCoordinator::new(frame_limit, aggregate_limit).unwrap();
                match runtime.start_unvalidated(factory) {
                    Ok(_) => {
                        let receiver = runtime.claim_receiver().unwrap();
                        return Self {
                            runtime,
                            receiver,
                            requests,
                            _capture_serial: capture_serial,
                        };
                    }
                    Err(CaptureStartError::OwnerGate(
                        crate::capture::CaptureOwnerGateError::AlreadyOwned { .. },
                    )) if Instant::now() < deadline => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("could not start capture test source: {error}"),
                }
            }
        }

        fn next_frame(&mut self) -> OwnedCaptureFrame {
            self.requests.send(()).unwrap();
            match self
                .receiver
                .receive_owned_with_timeout(Duration::from_secs(1))
            {
                CaptureReceiveOutcome::Frame(frame) => frame,
                other => panic!("expected captured frame, got {other:?}"),
            }
        }
    }

    impl Drop for CaptureHarness {
        fn drop(&mut self) {
            let _ = self.runtime.stop(Duration::from_secs(1));
        }
    }

    fn draw_line(pixels: &mut [u8], from: [f64; 2], to: [f64; 2]) {
        let dx = to[0] - from[0];
        let dy = to[1] - from[1];
        let length_squared = dx * dx + dy * dy;
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                let t = (((x as f64 - from[0]) * dx + (y as f64 - from[1]) * dy) / length_squared)
                    .clamp(0.0, 1.0);
                let nearest = [from[0] + t * dx, from[1] + t * dy];
                if (x as f64 - nearest[0]).hypot(y as f64 - nearest[1]) <= 2.5 {
                    let offset = (y * WIDTH as usize + x) * 3;
                    pixels[offset..offset + 3].copy_from_slice(&[235, 235, 235]);
                }
            }
        }
    }

    fn test_config() -> CueGeometryConfig {
        CueGeometryConfig {
            max_width_px: WIDTH,
            max_height_px: HEIGHT,
            max_pixels: (WIDTH * HEIGHT) as usize,
            max_edge_samples: 12_000,
            max_point_visits: 300_000,
            max_temporary_bytes: 12_000 * std::mem::size_of::<[f64; 2]>() * 2,
            ..CueGeometryConfig::default()
        }
    }

    fn tag(frame: &OwnedCaptureFrame) -> CueWorkTag {
        CueWorkTag {
            source_session_id: frame.source().session_id.clone(),
            frame_id: frame.source().frame_id.clone(),
            configuration_revision: 7,
            calibration_revision: 11,
            motion_generation: 13,
        }
    }

    fn poll_until_ready(worker: &mut CueGeometryWorker) -> super::CueGeometryCompletion {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(completion) = worker.poll_result() {
                return completion;
            }
            assert!(Instant::now() < deadline, "cue worker did not complete");
            thread::yield_now();
        }
    }

    #[test]
    fn returns_geometry_with_exact_tag_frame_lease_and_observation_timing() {
        let _serial = CAPTURE_RUNTIME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut capture = CaptureHarness::start();
        let frame = capture.next_frame();
        let expected_tag = tag(&frame);
        let mut worker = CueGeometryWorker::spawn(test_config()).unwrap();

        worker
            .try_submit(expected_tag.clone(), frame, &[], Duration::from_millis(23))
            .unwrap();
        let completion = poll_until_ready(&mut worker);
        let geometry = completion.result.unwrap().expect("cue geometry should fit");

        assert_eq!(completion.tag, expected_tag);
        assert_eq!(completion.frame.source().frame_id, expected_tag.frame_id);
        assert!(geometry.confidence > 0.5);
        assert!(completion.timing.processing_started_at >= completion.timing.submitted_at);
        assert!(
            completion.timing.processing_completed_at >= completion.timing.processing_started_at
        );
        assert!(completion.timing.observation_age_at_completion >= Duration::from_millis(23));
        assert_eq!(
            geometry.observation_age_ms,
            completion.timing.observation_age_at_completion.as_millis() as u64
        );
        assert_eq!(
            worker.shutdown(Duration::from_secs(1)),
            CueGeometryWorkerShutdown::Stopped
        );
    }

    #[test]
    fn refuses_a_second_frame_without_queueing_and_returns_its_lease() {
        let _serial = CAPTURE_RUNTIME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut capture = CaptureHarness::start();
        let first = capture.next_frame();
        let second = capture.next_frame();
        let second_tag = tag(&second);
        let (entered_sender, entered_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let mut worker = CueGeometryWorker::spawn_inner(test_config(), move |_, _, _, _, _, _| {
            let _ = entered_sender.send(());
            let _ = release_receiver.recv();
            Ok(None)
        })
        .unwrap();
        worker
            .try_submit(tag(&first), first, &[], Duration::ZERO)
            .unwrap();
        entered_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let error = worker
            .try_submit(second_tag.clone(), second, &[], Duration::ZERO)
            .unwrap_err();
        assert_eq!(error.reason(), CueGeometrySubmitRefusal::Busy);
        let returned = error.into_frame();
        assert_eq!(returned.source().frame_id, second_tag.frame_id);
        assert_eq!(returned.source().session_id, second_tag.source_session_id);

        release_sender.send(()).unwrap();
        let completion = poll_until_ready(&mut worker);
        assert_eq!(completion.frame.source().frame_id, "frame-1");
        assert_eq!(
            worker.shutdown(Duration::from_secs(1)),
            CueGeometryWorkerShutdown::Stopped
        );
    }

    #[test]
    fn rejects_a_tag_that_does_not_identify_the_owned_frame() {
        let _serial = CAPTURE_RUNTIME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut capture = CaptureHarness::start();
        let frame = capture.next_frame();
        let mut mismatched = tag(&frame);
        mismatched.frame_id = "different-frame".to_owned();
        let mut worker = CueGeometryWorker::spawn(test_config()).unwrap();

        let error = worker
            .try_submit(mismatched, frame, &[], Duration::ZERO)
            .unwrap_err();
        assert_eq!(
            error.reason(),
            CueGeometrySubmitRefusal::TagDoesNotMatchFrame
        );
        let returned = error.into_frame();
        assert_eq!(returned.source().frame_id, "frame-1");
        assert!(worker.can_accept_work());
        assert_eq!(
            worker.shutdown(Duration::from_secs(1)),
            CueGeometryWorkerShutdown::Stopped
        );
    }

    #[test]
    fn estimator_panic_is_captured_as_a_per_frame_failure() {
        let _serial = CAPTURE_RUNTIME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut capture = CaptureHarness::start();
        let frame = capture.next_frame();
        let mut worker = CueGeometryWorker::spawn_inner(test_config(), |_, _, _, _, _, _| {
            panic!("injected cue estimator panic")
        })
        .unwrap();

        worker
            .try_submit(tag(&frame), frame, &[], Duration::ZERO)
            .unwrap();
        let completion = poll_until_ready(&mut worker);
        let failure = completion.result.unwrap_err();
        assert_eq!(
            failure.kind,
            CueGeometryWorkerFailureKind::EstimatorPanicked
        );
        assert!(failure.message.contains("injected cue estimator panic"));
        assert_eq!(completion.frame.source().frame_id, "frame-1");
        assert_eq!(
            worker.shutdown(Duration::from_secs(1)),
            CueGeometryWorkerShutdown::Stopped
        );
    }

    #[test]
    fn shutdown_is_bounded_while_an_estimator_is_blocked() {
        let _serial = CAPTURE_RUNTIME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut capture = CaptureHarness::start();
        let frame = capture.next_frame();
        let (entered_sender, entered_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let mut worker = CueGeometryWorker::spawn_inner(test_config(), move |_, _, _, _, _, _| {
            let _ = entered_sender.send(());
            let _ = release_receiver.recv();
            Ok(None)
        })
        .unwrap();
        worker
            .try_submit(tag(&frame), frame, &[], Duration::ZERO)
            .unwrap();
        entered_receiver
            .recv_timeout(Duration::from_secs(1))
            .unwrap();

        let started = Instant::now();
        assert_eq!(
            worker.shutdown(Duration::from_millis(20)),
            CueGeometryWorkerShutdown::TimedOutDetached
        );
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(worker.state(), CueGeometryWorkerState::ShuttingDown);

        release_sender.send(()).unwrap();
        assert_eq!(
            worker.shutdown(Duration::from_secs(1)),
            CueGeometryWorkerShutdown::Stopped
        );
        assert!(worker.poll_result().is_some());
    }

    #[test]
    fn proposal_count_is_bounded_before_copying() {
        let _serial = CAPTURE_RUNTIME_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut capture = CaptureHarness::start();
        let frame = capture.next_frame();
        let expected_tag = tag(&frame);
        let config = CueGeometryConfig {
            max_proposals: 1,
            ..test_config()
        };
        let mut worker = CueGeometryWorker::spawn(config).unwrap();
        let proposals = [
            CueProposal {
                left: 0,
                top: 0,
                right: 10,
                bottom: 10,
            },
            CueProposal {
                left: 10,
                top: 10,
                right: 20,
                bottom: 20,
            },
        ];

        let error = worker
            .try_submit(expected_tag.clone(), frame, &proposals, Duration::ZERO)
            .unwrap_err();
        match error {
            CueGeometrySubmitError::TooManyProposals {
                frame,
                maximum,
                actual,
            } => {
                assert_eq!(maximum, 1);
                assert_eq!(actual, 2);
                assert_eq!(frame.source().frame_id, expected_tag.frame_id);
            }
            other => panic!("expected bounded proposal refusal, got {other:?}"),
        }
        assert_eq!(
            worker.shutdown(Duration::from_secs(1)),
            CueGeometryWorkerShutdown::Stopped
        );
    }
}
