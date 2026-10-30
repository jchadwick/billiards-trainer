//! Single-flight detector ownership and bounded shutdown coverage.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use billiards_perception::capture::{
    CaptureAggregateFrameLimit, CaptureAppliedCameraControls, CaptureAppliedFrameGeometry,
    CaptureCapability, CaptureControlReadback, CaptureFrameBudget, CaptureFrameLimit,
    CaptureFrameRate, CaptureInterrupt, CaptureOpenError, CapturePixelFormat, CaptureReadOutcome,
    CaptureReceiveOutcome, CaptureRuntimeCoordinator, CaptureSource, CaptureSourceCapabilities,
    CaptureSourceFactory, CaptureSourceMetadata, CaptureTimestampSemantics, CaptureVideoProfile,
    OwnedCaptureFrame, RgbFrameEnvelope,
};
use billiards_perception::capture_pipeline::Rgb8FrameDetector;
use billiards_perception::detection_batch::{InferenceSettings, ModelIdentity};
use billiards_perception::detector_worker::{
    DetectorCompletion, DetectorSubmitRefusal, DetectorWorkTag, DetectorWorker,
    DetectorWorkerFailureKind, DetectorWorkerShutdown, DetectorWorkerState,
};
use billiards_perception::inference::{FrameInferenceOutput, FrameInferenceTimings};
use billiards_perception::phase_scheduler::{
    CueEvidence, ImagePoint, LayoutObservation, MotionEvidence, Phase, PhaseScheduler,
    SchedulerConfig,
};
use billiards_perception::ImageSpace;

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
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
                Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(CaptureReadOutcome::End),
            }
        }

        self.frame_number += 1;
        let pixels = frame_budget
            .allocate_rgb8_buffer(WIDTH, HEIGHT, frame_limit)
            .map_err(|error| error.to_string())?;
        let frame = RgbFrameEnvelope::new(
            billiards_perception::detection_batch::SourceFrame {
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
    receiver: billiards_perception::capture::CaptureFrameReceiver,
    requests: mpsc::Sender<()>,
}

impl CaptureHarness {
    fn start() -> Self {
        let frame_limit = CaptureFrameLimit::new(FRAME_BYTES).unwrap();
        let aggregate_limit = CaptureAggregateFrameLimit::new(FRAME_BYTES * 8).unwrap();
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
            "worker-test-camera",
            None,
            "detector-worker-test",
            profile.clone(),
            profile,
            capabilities,
            CaptureAppliedFrameGeometry::new(1, "worker-test-profile", ImageSpace::Raw, 0).unwrap(),
            controls,
            CaptureTimestampSemantics::ReadCompletionOnly,
        )
        .unwrap();
        let interrupted = Arc::new(AtomicBool::new(false));
        let (requests, request_receiver) = mpsc::channel();
        let (wake, _wake_receiver) = mpsc::channel();
        let factory = TestFactory {
            metadata,
            interrupted,
            wake,
            requests: request_receiver,
        };
        let mut runtime = CaptureRuntimeCoordinator::new(frame_limit, aggregate_limit).unwrap();
        runtime.start_unvalidated(factory).unwrap();
        let receiver = runtime.claim_receiver().unwrap();
        Self {
            runtime,
            receiver,
            requests,
        }
    }

    fn next_frame(&mut self) -> OwnedCaptureFrame {
        self.requests.send(()).unwrap();
        match self
            .receiver
            .receive_owned_with_timeout(Duration::from_secs(1))
        {
            CaptureReceiveOutcome::Frame(frame) => frame,
            other => panic!("expected a captured frame, got {other:?}"),
        }
    }
}

impl Drop for CaptureHarness {
    fn drop(&mut self) {
        let _ = self.runtime.stop(Duration::from_secs(1));
    }
}

#[derive(Clone, Copy)]
enum DetectorBehavior {
    Error,
    BlockOnce,
}

struct TestDetector {
    behavior: DetectorBehavior,
    calls: Arc<AtomicUsize>,
    entered: Option<mpsc::Sender<()>>,
    release: Option<mpsc::Receiver<()>>,
    model: ModelIdentity,
    settings: InferenceSettings,
}

impl TestDetector {
    fn new(behavior: DetectorBehavior) -> Self {
        Self {
            behavior,
            calls: Arc::new(AtomicUsize::new(0)),
            entered: None,
            release: None,
            model: ModelIdentity {
                artifact_sha256: "0".repeat(64),
                source_checkpoint_sha256: None,
                classes: Vec::new(),
            },
            settings: InferenceSettings {
                input_width_px: WIDTH,
                input_height_px: HEIGHT,
                confidence_threshold: 0.5,
                iou_threshold: 0.5,
                max_detections: 16,
            },
        }
    }

    fn blocking() -> (Self, mpsc::Receiver<()>, mpsc::Sender<()>) {
        let (entered_sender, entered_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let mut detector = Self::new(DetectorBehavior::BlockOnce);
        detector.entered = Some(entered_sender);
        detector.release = Some(release_receiver);
        (detector, entered_receiver, release_sender)
    }
}

impl Rgb8FrameDetector for TestDetector {
    fn infer_rgb8(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels: &[u8],
    ) -> anyhow::Result<FrameInferenceOutput> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        anyhow::ensure!(pixels.len() == (width_px * height_px * 3) as usize);
        match self.behavior {
            DetectorBehavior::Error => anyhow::bail!("fixture inference failure"),
            DetectorBehavior::BlockOnce => {
                if let Some(entered) = self.entered.take() {
                    entered.send(()).unwrap();
                    self.release
                        .take()
                        .expect("first call has a release gate")
                        .recv()
                        .unwrap();
                }
            }
        }
        Ok(FrameInferenceOutput {
            source_width_px: width_px,
            source_height_px: height_px,
            detections: Vec::new(),
            timings: FrameInferenceTimings {
                preprocess_ms: 0.0,
                inference_ms: 0.0,
                decode_and_nms_ms: 0.0,
                engine_total_ms: 0.0,
            },
            ball_refinements: None,
        })
    }

    fn model_identity(&self) -> &ModelIdentity {
        &self.model
    }

    fn inference_settings(&self) -> &InferenceSettings {
        &self.settings
    }
}

fn tag(frame: &OwnedCaptureFrame, motion_generation: u64) -> DetectorWorkTag {
    DetectorWorkTag {
        source_session_id: frame.source().session_id.clone(),
        frame_id: frame.source().frame_id.clone(),
        configuration_revision: 3,
        calibration_revision: 7,
        motion_generation,
    }
}

fn wait_for_completion(worker: &mut DetectorWorker) -> DetectorCompletion {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(completion) = worker.poll_result() {
            return completion;
        }
        assert!(Instant::now() < deadline, "inference completion timed out");
        thread::yield_now();
    }
}

#[test]
fn worker_rejects_busy_dispatch_without_queue_and_returns_tagged_frame() {
    let _serial = CAPTURE_RUNTIME_LOCK.lock().unwrap();
    let mut capture = CaptureHarness::start();
    let (detector, entered, release) = TestDetector::blocking();
    let mut worker = DetectorWorker::spawn(detector).unwrap();
    let first_frame = capture.next_frame();
    let first_tag = tag(&first_frame, 11);
    worker
        .try_submit(first_tag.clone(), first_frame)
        .expect("first job rendezvous succeeds");
    entered
        .recv_timeout(Duration::from_secs(1))
        .expect("detector entered its blocked call");

    let second_frame = capture.next_frame();
    let second_error = worker
        .try_submit(tag(&second_frame, 12), second_frame)
        .expect_err("a busy worker must reject rather than queue a frame");
    assert_eq!(second_error.reason(), DetectorSubmitRefusal::Busy);
    let second_frame = second_error.into_frame();

    let third_frame = capture.next_frame();
    let third_error = worker
        .try_submit(tag(&third_frame, 13), third_frame)
        .expect_err("a second dispatch cannot accumulate behind the first");
    assert_eq!(third_error.reason(), DetectorSubmitRefusal::Busy);

    release.send(()).unwrap();
    let completion = wait_for_completion(&mut worker);
    assert_eq!(completion.tag, first_tag);
    assert_eq!(completion.tag.motion_generation, 11);
    assert_eq!(completion.frame.source().frame_id, "frame-1");
    assert!(completion.inference_completed >= completion.inference_started);
    assert!(completion.result.is_ok());

    // The completion slot itself is still the single outstanding result until
    // consumed; the next frame may be submitted only after polling it.
    let mut retry_frame = second_frame;
    loop {
        match worker.try_submit(tag(&retry_frame, 12), retry_frame) {
            Ok(()) => break,
            Err(error) if error.reason() == DetectorSubmitRefusal::Busy => {
                retry_frame = error.into_frame();
                thread::yield_now();
            }
            Err(error) => panic!("unexpected retry refusal: {:?}", error.reason()),
        }
    }
    let next_completion = wait_for_completion(&mut worker);
    assert_eq!(next_completion.tag.motion_generation, 12);
    assert!(next_completion.result.is_ok());
    assert_eq!(
        worker.shutdown(Duration::from_secs(1)),
        DetectorWorkerShutdown::Stopped
    );
}

#[test]
fn motion_invalidation_advances_while_detector_is_blocked_and_late_result_is_obsolete() {
    let _serial = CAPTURE_RUNTIME_LOCK.lock().unwrap();
    let mut capture = CaptureHarness::start();
    let (detector, entered, release) = TestDetector::blocking();
    let mut worker = DetectorWorker::spawn(detector).unwrap();
    let mut scheduler = PhaseScheduler::new(SchedulerConfig::default()).unwrap();
    for time_ms in [0, 150, 300, 450, 600, 750, 800] {
        scheduler
            .observe_motion(Duration::from_millis(time_ms), MotionEvidence::Quiet)
            .unwrap();
    }
    for time_ms in [800, 850, 900] {
        scheduler
            .observe_layout(
                Duration::from_millis(time_ms),
                LayoutObservation::Success(vec![ImagePoint { x: 10.0, y: 10.0 }]),
            )
            .unwrap();
    }
    scheduler
        .observe_cue(Duration::from_millis(910), CueEvidence::Visible)
        .unwrap();
    assert_eq!(scheduler.phase(), Phase::Aiming);
    let in_flight_generation = scheduler.motion_generation();
    let first_frame = capture.next_frame();
    let first_tag = tag(&first_frame, in_flight_generation);
    worker.try_submit(first_tag.clone(), first_frame).unwrap();
    entered
        .recv_timeout(Duration::from_secs(1))
        .expect("detector entered its blocked call");

    // This is the coordinator's independent lightweight update while the
    // inference thread remains blocked. Guidance is invalid before the model
    // returns, and the new generation makes that eventual result unusable.
    scheduler
        .observe_motion(Duration::from_millis(950), MotionEvidence::Moving)
        .unwrap();
    assert_eq!(scheduler.phase(), Phase::Shot);
    assert!(!scheduler.map_trusted());
    assert_ne!(scheduler.motion_generation(), in_flight_generation);

    let second_frame = capture.next_frame();
    let refusal = worker
        .try_submit(
            tag(&second_frame, scheduler.motion_generation()),
            second_frame,
        )
        .expect_err("motion processing must not queue work behind the blocked inference");
    assert_eq!(refusal.reason(), DetectorSubmitRefusal::Busy);

    release.send(()).unwrap();
    let completion = wait_for_completion(&mut worker);
    assert_eq!(completion.tag, first_tag);
    assert_ne!(
        completion.tag.motion_generation,
        scheduler.motion_generation()
    );
    assert!(completion.result.is_ok());
    assert_eq!(
        worker.shutdown(Duration::from_secs(1)),
        DetectorWorkerShutdown::Stopped
    );
}

#[test]
fn inference_errors_are_completed_with_the_original_context_and_lease() {
    let _serial = CAPTURE_RUNTIME_LOCK.lock().unwrap();
    let mut capture = CaptureHarness::start();
    let mut worker = DetectorWorker::spawn(TestDetector::new(DetectorBehavior::Error)).unwrap();
    let frame = capture.next_frame();
    let work_tag = tag(&frame, 29);
    worker.try_submit(work_tag.clone(), frame).unwrap();

    let completion = wait_for_completion(&mut worker);
    assert_eq!(completion.tag, work_tag);
    assert_eq!(completion.frame.source().frame_id, "frame-1");
    assert!(completion.inference_completed >= completion.inference_started);
    let failure = completion.result.unwrap_err();
    assert_eq!(failure.kind, DetectorWorkerFailureKind::Inference);
    assert!(failure.message.contains("fixture inference failure"));
    assert_eq!(
        worker.shutdown(Duration::from_secs(1)),
        DetectorWorkerShutdown::Stopped
    );
}

#[test]
fn shutdown_is_bounded_and_detaches_a_blocked_native_detector() {
    let _serial = CAPTURE_RUNTIME_LOCK.lock().unwrap();
    let mut capture = CaptureHarness::start();
    let (detector, entered, release) = TestDetector::blocking();
    let mut worker = DetectorWorker::spawn(detector).unwrap();
    let frame = capture.next_frame();
    worker.try_submit(tag(&frame, 41), frame).unwrap();
    entered
        .recv_timeout(Duration::from_secs(1))
        .expect("detector entered its blocked call");

    let started = Instant::now();
    assert_eq!(
        worker.shutdown(Duration::from_millis(40)),
        DetectorWorkerShutdown::TimedOutDetached
    );
    assert!(started.elapsed() < Duration::from_millis(500));

    // Unblock the detached fixture so the test process does not retain a live
    // thread. Production containment for an irrecoverably stuck native call is
    // the existing perception-process supervisor.
    release.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while worker.state() != DetectorWorkerState::Stopped {
        assert!(Instant::now() < deadline, "detached worker did not exit");
        thread::yield_now();
    }
    let completion = worker
        .poll_result()
        .expect("completed result remains pollable");
    assert_eq!(completion.tag.motion_generation, 41);
    assert!(completion.result.is_ok());
}
