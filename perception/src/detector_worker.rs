//! Single-flight owner for the mutable frame detector.
//!
//! A coordinator transfers one owned capture lease only when the worker is
//! waiting for work and has no unconsumed completion. The handoff is a
//! condition-variable rendezvous rather than a frame queue: a missed dispatch
//! opportunity leaves the frame with the caller and cannot accumulate work.
//! The worker thread owns the detector for its entire lifetime.

use std::any::Any;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::ball_localization::BallRefinementIdentity;
use crate::capture::OwnedCaptureFrame;
use crate::capture_pipeline::Rgb8FrameDetector;
use crate::inference::FrameInferenceOutput;

/// Revisions and identities that make an inference result safe to associate
/// with the context in which its pixels were captured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectorWorkTag {
    pub source_session_id: String,
    pub frame_id: String,
    pub configuration_revision: u64,
    pub calibration_revision: u64,
    pub motion_generation: u64,
}

/// The detector result and the exact owned frame lease from which it came.
///
/// Returning the lease, including on inference errors, lets the coordinator
/// retain correct aggregate-memory accounting until it has finished all
/// frame-local work.
#[derive(Debug)]
pub struct DetectorCompletion {
    pub tag: DetectorWorkTag,
    pub frame: OwnedCaptureFrame,
    /// Process-local monotonic bounds around the detector call.
    pub inference_started: Instant,
    pub inference_completed: Instant,
    pub result: Result<FrameInferenceOutput, DetectorWorkerFailure>,
}

/// A recoverable detector failure. Native runtime errors are converted to
/// their full anyhow display chain; a panic is captured so it cannot silently
/// kill the only inference thread while leaving the worker marked busy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectorWorkerFailure {
    pub kind: DetectorWorkerFailureKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorWorkerFailureKind {
    Inference,
    DetectorPanicked,
    WorkerPanicked,
}

impl Display for DetectorWorkerFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.kind, self.message)
    }
}

impl Display for DetectorWorkerFailureKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Inference => "detector inference failed",
            Self::DetectorPanicked => "detector inference panicked",
            Self::WorkerPanicked => "detector worker panicked",
        })
    }
}

/// Why a nonblocking dispatch was refused. The caller receives ownership of
/// the frame back and can continue lightweight processing before dropping it.
#[derive(Debug)]
pub enum DetectorSubmitError {
    Busy(OwnedCaptureFrame),
    ShuttingDown(OwnedCaptureFrame),
    WorkerUnavailable(OwnedCaptureFrame),
    TagDoesNotMatchFrame {
        frame: OwnedCaptureFrame,
        expected_session_id: String,
        expected_frame_id: String,
    },
}

impl DetectorSubmitError {
    /// Recover the lease rejected by this dispatch attempt.
    pub fn into_frame(self) -> OwnedCaptureFrame {
        match self {
            Self::Busy(frame)
            | Self::ShuttingDown(frame)
            | Self::WorkerUnavailable(frame)
            | Self::TagDoesNotMatchFrame { frame, .. } => frame,
        }
    }

    pub fn reason(&self) -> DetectorSubmitRefusal {
        match self {
            Self::Busy(_) => DetectorSubmitRefusal::Busy,
            Self::ShuttingDown(_) => DetectorSubmitRefusal::ShuttingDown,
            Self::WorkerUnavailable(_) => DetectorSubmitRefusal::WorkerUnavailable,
            Self::TagDoesNotMatchFrame { .. } => DetectorSubmitRefusal::TagDoesNotMatchFrame,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorSubmitRefusal {
    Busy,
    ShuttingDown,
    WorkerUnavailable,
    TagDoesNotMatchFrame,
}

/// Result of requesting a bounded worker shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorWorkerShutdown {
    Stopped,
    /// The inference thread did not stop within the supplied bound. Its join
    /// handle is detached; a process supervisor can contain a stuck native
    /// runtime by terminating this perception process.
    TimedOutDetached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorWorkerState {
    Ready,
    WorkerWaking,
    InferenceInFlight,
    ResultReady,
    ShuttingDown,
    Stopped,
    Failed,
}

struct InferenceJob {
    tag: DetectorWorkTag,
    frame: OwnedCaptureFrame,
}

struct WorkerState {
    job: Option<InferenceJob>,
    completion: Option<DetectorCompletion>,
    outstanding: bool,
    worker_waiting: bool,
    shutting_down: bool,
    worker_exited: bool,
    terminal_failure: Option<DetectorWorkerFailure>,
}

struct Shared {
    state: Mutex<WorkerState>,
    changed: Condvar,
}

/// Handle for the one detector-owning thread.
///
/// The type itself does not own the detector; only the worker closure does.
/// `Drop` requests shutdown and detaches instead of joining, so a native
/// inference that never returns cannot hang destruction of the runtime.
pub struct DetectorWorker {
    shared: Arc<Shared>,
    join: Option<JoinHandle<()>>,
    _not_generic: PhantomData<fn()>,
}

impl std::fmt::Debug for DetectorWorker {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DetectorWorker")
            .field("state", &self.state())
            .field(
                "thread_finished",
                &self.join.as_ref().is_none_or(JoinHandle::is_finished),
            )
            .finish()
    }
}

impl DetectorWorker {
    /// Start the dedicated thread and wait until it is ready for a direct
    /// rendezvous. The detector is moved into that thread and never exposed to
    /// the coordinator again.
    pub fn spawn<D>(detector: D) -> std::io::Result<Self>
    where
        D: Rgb8FrameDetector + Send + 'static,
    {
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
        let join = thread::Builder::new()
            .name("pool-ball-detector".to_owned())
            .spawn(move || run_worker(detector, worker_shared))?;

        // Do not make the first scheduled inference lose a frame simply
        // because the OS has not started the new thread yet.
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
            _not_generic: PhantomData,
        })
    }

    /// Attempt to transfer one frame without waiting. There is no pending
    /// frame queue: success requires that the worker is already waiting, and
    /// every refusal returns the original lease to the caller.
    // The large error intentionally returns ownership of the input frame so
    // bounded overload handling never loses or clones its memory lease.
    #[allow(clippy::result_large_err)]
    pub fn try_submit(
        &mut self,
        tag: DetectorWorkTag,
        frame: OwnedCaptureFrame,
    ) -> Result<(), DetectorSubmitError> {
        let frame_source = frame.source();
        if frame_source.session_id != tag.source_session_id || frame_source.frame_id != tag.frame_id
        {
            return Err(DetectorSubmitError::TagDoesNotMatchFrame {
                frame,
                expected_session_id: tag.source_session_id,
                expected_frame_id: tag.frame_id,
            });
        }

        let mut state = lock_state(&self.shared);
        if state.shutting_down {
            return Err(DetectorSubmitError::ShuttingDown(frame));
        }
        if state.worker_exited || state.terminal_failure.is_some() {
            return Err(DetectorSubmitError::WorkerUnavailable(frame));
        }
        if state.outstanding || !state.worker_waiting {
            return Err(DetectorSubmitError::Busy(frame));
        }

        state.outstanding = true;
        state.worker_waiting = false;
        state.job = Some(InferenceJob { tag, frame });
        self.shared.changed.notify_one();
        Ok(())
    }

    /// Take a completed inference, if available. Until the completion is
    /// consumed, the worker remains single-flight and new submissions fail as
    /// busy, preventing a result slot from becoming an implicit queue.
    pub fn poll_result(&mut self) -> Option<DetectorCompletion> {
        let mut state = lock_state(&self.shared);
        let completion = state.completion.take()?;
        state.outstanding = false;
        self.shared.changed.notify_all();
        Some(completion)
    }

    /// Current worker state for diagnostics and coordinator scheduling.
    pub fn state(&self) -> DetectorWorkerState {
        let state = lock_state(&self.shared);
        if state.terminal_failure.is_some() {
            DetectorWorkerState::Failed
        } else if state.worker_exited {
            DetectorWorkerState::Stopped
        } else if state.shutting_down {
            DetectorWorkerState::ShuttingDown
        } else if state.completion.is_some() {
            DetectorWorkerState::ResultReady
        } else if state.outstanding {
            DetectorWorkerState::InferenceInFlight
        } else if state.worker_waiting {
            DetectorWorkerState::Ready
        } else {
            DetectorWorkerState::WorkerWaking
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

    /// Report a failure outside a normal per-frame detector return value.
    pub fn terminal_failure(&self) -> Option<DetectorWorkerFailure> {
        lock_state(&self.shared).terminal_failure.clone()
    }

    /// Request shutdown and wait no longer than `timeout`. If native inference
    /// is stuck, the join handle is detached and this call returns so the
    /// parent process supervisor can terminate the process.
    pub fn shutdown(&mut self, timeout: Duration) -> DetectorWorkerShutdown {
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
                self.join.take(); // Dropping JoinHandle explicitly detaches.
                return DetectorWorkerShutdown::TimedOutDetached;
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
                return DetectorWorkerShutdown::TimedOutDetached;
            }
        }
        drop(state);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        DetectorWorkerShutdown::Stopped
    }
}

impl Drop for DetectorWorker {
    fn drop(&mut self) {
        {
            let mut state = lock_state(&self.shared);
            state.shutting_down = true;
            self.shared.changed.notify_all();
        }
        // Dropping a JoinHandle detaches. Never let an unresponsive native
        // detector turn an ordinary owner drop into an unbounded join.
        self.join.take();
    }
}

fn run_worker<D: Rgb8FrameDetector>(mut detector: D, shared: Arc<Shared>) {
    let result = catch_unwind(AssertUnwindSafe(|| worker_loop(&mut detector, &shared)));
    let mut state = lock_state(&shared);
    if let Err(payload) = result {
        state.terminal_failure = Some(DetectorWorkerFailure {
            kind: DetectorWorkerFailureKind::WorkerPanicked,
            message: panic_payload_message(payload),
        });
    }
    state.worker_waiting = false;
    state.worker_exited = true;
    shared.changed.notify_all();
}

fn worker_loop<D: Rgb8FrameDetector>(detector: &mut D, shared: &Arc<Shared>) {
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

        let inference_started = Instant::now();
        let inference = catch_unwind(AssertUnwindSafe(|| {
            detector.infer_rgb8(
                job.frame.source().width_px,
                job.frame.source().height_px,
                job.frame.rgb8(),
            )
        }));
        let inference_completed = Instant::now();
        let result = match inference {
            Ok(Ok(mut output)) => {
                if let Some(refinements) = output.ball_refinements.as_mut() {
                    refinements.identity = Some(BallRefinementIdentity {
                        source_session_id: job.tag.source_session_id.clone(),
                        frame_id: job.tag.frame_id.clone(),
                        configuration_revision: job.tag.configuration_revision,
                        calibration_revision: job.tag.calibration_revision,
                        motion_generation: job.tag.motion_generation,
                    });
                }
                Ok(output)
            }
            Ok(Err(error)) => Err(DetectorWorkerFailure {
                kind: DetectorWorkerFailureKind::Inference,
                message: format!("{error:#}"),
            }),
            Err(payload) => Err(DetectorWorkerFailure {
                kind: DetectorWorkerFailureKind::DetectorPanicked,
                message: panic_payload_message(payload),
            }),
        };
        let completion = DetectorCompletion {
            tag: job.tag,
            frame: job.frame,
            inference_started,
            inference_completed,
            result,
        };

        let mut state = lock_state(shared);
        // There can be no second completion because the coordinator cannot
        // submit while `outstanding` remains true.
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

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}
