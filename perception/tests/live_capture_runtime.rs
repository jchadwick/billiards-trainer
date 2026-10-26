//! End-to-end tests for the Rust-owned live runtime using an injected source
//! and the production protocol-v3 scene transport.

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use billiards_perception::capture::CaptureReceiveOutcome;
use billiards_perception::capture::{
    CaptureAppliedCameraControls, CaptureAppliedFrameGeometry, CaptureCapability,
    CaptureControlReadback, CaptureExposureMode, CaptureFrameBudget, CaptureFrameLimit,
    CaptureFrameRate, CaptureInterrupt, CaptureOpenError, CaptureOwnerStatus, CapturePixelFormat,
    CaptureReadOutcome, CaptureSource, CaptureSourceCapabilities, CaptureSourceFactory,
    CaptureSourceMetadata, CaptureTimestampSemantics, CaptureVideoProfile, RgbFrameEnvelope,
};
use billiards_perception::capture_pipeline::{
    CapturePipelineConfig, CapturePipelineOutcome, Rgb8FrameDetector,
};
use billiards_perception::capture_runtime::{LiveCaptureLifecycle, LiveCaptureRuntime};
use billiards_perception::capture_settings_wire::{
    parse_capture_settings_json, CaptureSettingsSnapshotV1,
};
use billiards_perception::detection_batch::{
    ImageSpace as BatchImageSpace, InferenceSettings, ModelClass, ModelIdentity, SourceFrame,
};
use billiards_perception::framing::{read_framed_message, write_framed_message};
use billiards_perception::inference::{Detection, FrameInferenceOutput, FrameInferenceTimings};
use billiards_perception::phase_scheduler::{PhaseCadence, SchedulerConfig};
use billiards_perception::tracking::TrackingConfigV1;
use billiards_perception::SceneTransportConfig;
use serde_json::{json, Value};

const SMALL_FRAME_BYTES: usize = 16 * 16 * 3;
const MAXIMUM_MESSAGE_BYTES: usize = 64 * 1024;
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct TestDetector {
    model: ModelIdentity,
    settings: InferenceSettings,
    calls: Arc<AtomicUsize>,
    first_call_release: Option<mpsc::Receiver<()>>,
    shifting_cue_detection: bool,
}

impl Rgb8FrameDetector for TestDetector {
    fn infer_rgb8(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels: &[u8],
    ) -> anyhow::Result<FrameInferenceOutput> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        if call_index == 0 {
            if let Some(release) = self.first_call_release.as_ref() {
                release.recv().map_err(|error| anyhow::anyhow!(error))?;
            }
        }
        anyhow::ensure!(pixels.len() == (width_px * height_px * 3) as usize);
        let mut detections = vec![Detection {
            class_id: 1,
            class_name: "ball".to_owned(),
            confidence: 0.95,
            bbox_xyxy_px: [3.0, 4.0, 9.0, 10.0],
            center_xy_px: [6.0, 7.0],
        }];
        if self.shifting_cue_detection {
            // The cue moves between completed inferences while the ball stays
            // fixed. Cue detections must not affect ball-layout consensus.
            let cue_center_x = if call_index.is_multiple_of(2) {
                2.0
            } else {
                14.0
            };
            detections.push(Detection {
                class_id: 0,
                class_name: "cue".to_owned(),
                confidence: 0.9,
                bbox_xyxy_px: [cue_center_x - 1.0, 6.0, cue_center_x + 1.0, 8.0],
                center_xy_px: [cue_center_x, 7.0],
            });
        }
        Ok(FrameInferenceOutput {
            source_width_px: width_px,
            source_height_px: height_px,
            detections,
            timings: FrameInferenceTimings {
                preprocess_ms: 0.1,
                inference_ms: 0.2,
                decode_and_nms_ms: 0.1,
                engine_total_ms: 0.4,
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

struct TestInterrupt(Arc<AtomicBool>);

impl CaptureInterrupt for TestInterrupt {
    fn interrupt(&self) {
        self.0.store(true, Ordering::Release);
    }
}

struct TestFactory {
    metadata: CaptureSourceMetadata,
    interrupted: Arc<AtomicBool>,
    releases: Arc<AtomicUsize>,
    opens: Arc<AtomicUsize>,
    frame_count: usize,
    frame_interval: Duration,
}

impl CaptureSourceFactory for TestFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::clone(&self.interrupted)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(TestSource {
            metadata: self.metadata,
            interrupted: self.interrupted,
            releases: self.releases,
            emitted: 0,
            frame_count: self.frame_count,
            frame_interval: self.frame_interval,
        }))
    }
}

struct TestSource {
    metadata: CaptureSourceMetadata,
    interrupted: Arc<AtomicBool>,
    releases: Arc<AtomicUsize>,
    emitted: usize,
    frame_count: usize,
    frame_interval: Duration,
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
        if self.interrupted.load(Ordering::Acquire) || self.emitted >= self.frame_count {
            return Ok(CaptureReadOutcome::End);
        }
        if self.emitted > 0 && !self.frame_interval.is_zero() {
            thread::sleep(self.frame_interval);
        }
        self.emitted += 1;
        let geometry = self.metadata.output_frame_geometry().unwrap_or_else(|| {
            let profile = self.metadata.negotiated_profile();
            billiards_perception::capture::CaptureOutputFrameGeometry::new(
                profile.width_px(),
                profile.height_px(),
            )
            .unwrap()
        });
        let width_px = geometry.width_px();
        let height_px = geometry.height_px();
        let mut pixels = frame_budget
            .allocate_rgb8_buffer(width_px, height_px, frame_limit)
            .map_err(|error| error.to_string())?;
        pixels.as_mut_slice().fill(127);
        let frame = RgbFrameEnvelope::new(
            SourceFrame {
                session_id: session_id.to_owned(),
                frame_id: format!("frame-{}", self.emitted),
                timestamp: None,
                width_px,
                height_px,
            },
            pixels,
            std::time::Instant::now(),
            frame_limit,
        )
        .map_err(|error| error.to_string())?;
        Ok(CaptureReadOutcome::Frame(frame))
    }

    fn release(&mut self) -> Result<(), String> {
        self.releases.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct BlockingOpenFactory {
    metadata: CaptureSourceMetadata,
    interrupted: Arc<AtomicBool>,
    started: mpsc::Sender<()>,
    resume: mpsc::Receiver<()>,
}

impl CaptureSourceFactory for BlockingOpenFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::clone(&self.interrupted)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        self.started.send(()).unwrap();
        self.resume.recv().unwrap();
        if self.interrupted.load(Ordering::Acquire) {
            return Err(CaptureOpenError::Interrupted);
        }
        Ok(Box::new(TestSource {
            metadata: self.metadata,
            interrupted: self.interrupted,
            releases: Arc::new(AtomicUsize::new(0)),
            emitted: 0,
            frame_count: 0,
            frame_interval: Duration::ZERO,
        }))
    }
}

fn request_json() -> String {
    let mut request: Value = serde_json::from_str(include_str!(
        "../contracts/capture-settings/v1/capture-settings-default.json"
    ))
    .unwrap();
    for profile_name in ["requested_profile", "expected_negotiated_profile"] {
        request["requested"][profile_name]["width_px"] = json!(16);
        request["requested"][profile_name]["height_px"] = json!(16);
    }
    request["requested"]["maximum_frame_age_ms"] = json!(1_000);
    request["requested"]["maximum_rgb_frame_bytes"] = json!(SMALL_FRAME_BYTES);
    request["requested"]["maximum_aggregate_rgb_frame_bytes"] = json!(SMALL_FRAME_BYTES * 2);
    serde_json::to_string(&request).unwrap()
}

fn request_v2_json() -> String {
    let mut request: Value = serde_json::from_str(include_str!(
        "../contracts/capture-settings/v2/capture-settings-default.json"
    ))
    .unwrap();
    for profile_name in ["requested_profile", "expected_negotiated_profile"] {
        request["requested"][profile_name]["width_px"] = json!(16);
        request["requested"][profile_name]["height_px"] = json!(16);
    }
    request["requested"]["output_frame_geometry"]["width_px"] = json!(12);
    request["requested"]["output_frame_geometry"]["height_px"] = json!(12);
    request["requested"]["maximum_frame_age_ms"] = json!(1_000);
    request["requested"]["maximum_rgb_frame_bytes"] = json!(SMALL_FRAME_BYTES);
    request["requested"]["maximum_aggregate_rgb_frame_bytes"] = json!(SMALL_FRAME_BYTES * 2);
    serde_json::to_string(&request).unwrap()
}

fn model_and_calls() -> (TestDetector, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let detector = TestDetector {
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
        calls: Arc::clone(&calls),
        first_call_release: None,
        shifting_cue_detection: false,
    };
    (detector, calls)
}

fn pipeline_config(
    address: SocketAddr,
    revision: u64,
    maximum_frame_age: Duration,
) -> CapturePipelineConfig {
    CapturePipelineConfig {
        engine_instance_id: "rust-live-runtime-test".to_owned(),
        capture_configuration_revision: revision,
        image_space: BatchImageSpace::Raw,
        maximum_frame_age,
        phase_schedule: None,
        calibration: None,
        scene_transport: SceneTransportConfig {
            address,
            maximum_message_bytes: MAXIMUM_MESSAGE_BYTES,
            public_state_delivery_enabled: false,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
            reconnect_initial_delay: Duration::from_millis(10),
            reconnect_maximum_delay: Duration::from_millis(20),
        },
    }
}

fn tracking(camera_profile_id: &str, rectification_revision: u64) -> TrackingConfigV1 {
    TrackingConfigV1::new(
        1,
        Some(0),
        camera_profile_id,
        40.0,
        Duration::from_millis(100),
        Duration::from_secs(1),
        Duration::from_secs(3),
        32,
        8,
        1,
        rectification_revision,
    )
    .unwrap()
}

fn metadata(
    settings: &billiards_perception::live_config::ExpectedCaptureSettings,
) -> CaptureSourceMetadata {
    let requested_profile = settings.requested_profile().clone();
    let negotiated_profile = settings.expected_negotiated_profile().clone();
    let format = negotiated_profile.pixel_format().clone();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![format]),
        CaptureCapability::Unsupported,
        CaptureCapability::Supported,
        CaptureCapability::Supported,
        CaptureCapability::Unsupported,
    )
    .unwrap();
    let controls = CaptureAppliedCameraControls::new(
        CaptureControlReadback::Applied(CaptureExposureMode::Automatic),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Applied(1.0),
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
        CaptureControlReadback::Unknown,
    )
    .unwrap();
    let source_metadata = CaptureSourceMetadata::new(
        settings.device_id(),
        Some("deterministic source".to_owned()),
        "integration-test",
        requested_profile,
        negotiated_profile,
        capabilities,
        CaptureAppliedFrameGeometry::new(
            settings.configuration_revision(),
            settings.camera_profile_id(),
            settings.image_space(),
            settings.rectification_revision(),
        )
        .unwrap(),
        controls,
        CaptureTimestampSemantics::ReadCompletionOnly,
    )
    .unwrap();
    match settings.output_frame_geometry() {
        Some(geometry) => source_metadata.with_output_frame_geometry(geometry),
        None => source_metadata,
    }
}

fn runtime(
    request: &str,
    address: SocketAddr,
    metadata_override: Option<CaptureSourceMetadata>,
    frame_count: usize,
) -> (
    LiveCaptureRuntime<TestDetector>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    runtime_with_session_id(request, address, metadata_override, None, frame_count)
}

fn runtime_with_session_id(
    request: &str,
    address: SocketAddr,
    metadata_override: Option<CaptureSourceMetadata>,
    session_id: Option<&str>,
    frame_count: usize,
) -> (
    LiveCaptureRuntime<TestDetector>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
) {
    let settings = parse_capture_settings_json(request).unwrap();
    let source_metadata = metadata_override.unwrap_or_else(|| metadata(&settings.expected));
    let interrupted = Arc::new(AtomicBool::new(false));
    let releases = Arc::new(AtomicUsize::new(0));
    let opens = Arc::new(AtomicUsize::new(0));
    let (detector, calls) = model_and_calls();
    let config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    let factory = TestFactory {
        metadata: source_metadata,
        interrupted,
        releases: Arc::clone(&releases),
        opens: Arc::clone(&opens),
        frame_count,
        frame_interval: Duration::ZERO,
    };
    let track_settings = tracking(
        settings.expected.camera_profile_id(),
        settings.expected.rectification_revision(),
    );
    let runtime = match session_id {
        Some(session_id) => LiveCaptureRuntime::start_with_session_id(
            request,
            session_id,
            detector,
            track_settings,
            config,
            factory,
        ),
        None => LiveCaptureRuntime::start(request, detector, track_settings, config, factory),
    }
    .unwrap();
    (runtime, calls, releases, opens)
}

fn read_message(stream: &mut TcpStream) -> Value {
    let bytes = read_framed_message(stream, MAXIMUM_MESSAGE_BYTES).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn write_message(stream: &mut TcpStream, message: &Value) {
    let bytes = serde_json::to_vec(message).unwrap();
    write_framed_message(stream, &bytes, MAXIMUM_MESSAGE_BYTES).unwrap();
    stream.flush().unwrap();
}

fn receive_live_scene(listener: TcpListener) -> (Value, Value) {
    let (mut stream, _) = listener.accept().unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let hello = read_message(&mut stream);
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["protocol_version"], 3);
    let acknowledgement = &hello["capture_session_ack"];
    write_message(
        &mut stream,
        &json!({
            "type": "ready",
            "protocol_version": 3,
            "configuration_revision": acknowledgement["configuration_revision"],
            "request_sha256": acknowledgement["request_sha256"],
            "last_transport_sequence": null
        }),
    );
    let scene = read_message(&mut stream);
    (hello, scene)
}

fn acknowledge_and_drain_live_session(listener: TcpListener) {
    let (mut stream, _) = listener.accept().unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let hello = read_message(&mut stream);
    let acknowledgement = &hello["capture_session_ack"];
    write_message(
        &mut stream,
        &json!({
            "type": "ready",
            "protocol_version": 3,
            "configuration_revision": acknowledgement["configuration_revision"],
            "request_sha256": acknowledgement["request_sha256"],
            "last_transport_sequence": null
        }),
    );
    while read_framed_message(&mut stream, MAXIMUM_MESSAGE_BYTES).is_ok() {}
}

fn phase_test_schedule(
    ball_hz: f64,
    motion_hz: f64,
    quiet_dwell: Duration,
    layout_consensus_count: usize,
) -> SchedulerConfig {
    let cadence = PhaseCadence {
        motion_hz,
        ball_hz,
        cue_hz: 0.0,
    };
    SchedulerConfig {
        recovering: cadence,
        pre_shot: cadence,
        aiming: cadence,
        shot: PhaseCadence {
            motion_hz,
            ball_hz: 0.0,
            cue_hz: 0.0,
        },
        quiet_dwell,
        layout_consensus_count,
        layout_position_tolerance_px: 4.0,
        motion_evidence_expiry: Duration::from_secs(1),
        cue_evidence_expiry: Duration::from_millis(500),
        layout_evidence_expiry: Duration::from_secs(2),
    }
}

fn phase_runtime(
    request: &str,
    address: SocketAddr,
    detector: TestDetector,
    frame_count: usize,
    frame_interval: Duration,
    schedule: SchedulerConfig,
) -> LiveCaptureRuntime<TestDetector> {
    let settings = parse_capture_settings_json(request).unwrap();
    let interrupted = Arc::new(AtomicBool::new(false));
    let releases = Arc::new(AtomicUsize::new(0));
    let opens = Arc::new(AtomicUsize::new(0));
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(schedule);
    config.scene_transport.public_state_delivery_enabled = true;
    LiveCaptureRuntime::start(
        request,
        detector,
        tracking(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
        ),
        config,
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted,
            releases,
            opens,
            frame_count,
            frame_interval,
        },
    )
    .unwrap()
}

fn wait_for_detector_calls(calls: &AtomicUsize, minimum: usize) {
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while calls.load(Ordering::SeqCst) < minimum && std::time::Instant::now() < deadline {
        thread::yield_now();
    }
    assert!(calls.load(Ordering::SeqCst) >= minimum);
}

#[test]
fn cue_detections_do_not_change_ball_layout_settlement_consensus() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || acknowledge_and_drain_live_session(listener));
    let (mut detector, calls) = model_and_calls();
    detector.shifting_cue_detection = true;
    let mut runtime = phase_runtime(
        &request,
        address,
        detector,
        10,
        Duration::from_millis(60),
        phase_test_schedule(10.0, 20.0, Duration::from_millis(1), 2),
    );
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    for _ in 0..10 {
        match runtime.receive_one(Duration::from_secs(2)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) => {}
            CaptureReceiveOutcome::TimedOut => {
                panic!("scripted source did not emit its next frame")
            }
            CaptureReceiveOutcome::Closed => break,
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if runtime
            .scheduler_evidence()
            .unwrap()
            .is_some_and(|evidence| evidence.map_trusted)
        {
            break;
        }
    }

    wait_for_detector_calls(&calls, 2);
    let evidence = runtime.scheduler_evidence().unwrap().unwrap();
    assert!(
        evidence.map_trusted,
        "moving cue detections must not reset otherwise consistent ball consensus: {evidence:?}"
    );
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
}

#[test]
fn scheduled_cue_work_runs_on_a_single_flight_worker_and_reports_completion() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || acknowledge_and_drain_live_session(listener));
    let (detector, _) = model_and_calls();
    let mut schedule = phase_test_schedule(10.0, 30.0, Duration::from_millis(1), 2);
    schedule.recovering.cue_hz = 30.0;
    schedule.pre_shot.cue_hz = 30.0;
    schedule.aiming.cue_hz = 30.0;
    let mut runtime = phase_runtime(
        &request,
        address,
        detector,
        8,
        Duration::from_millis(40),
        schedule,
    );
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    for _ in 0..6 {
        match runtime.receive_one(Duration::from_secs(2)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) => {}
            CaptureReceiveOutcome::TimedOut => {
                panic!("scripted source did not emit its next frame")
            }
            other => panic!("unexpected live receive result: {other:?}"),
        }
    }
    let stats = runtime.phase_stats();
    assert!(
        stats.cue_dispatches > 0,
        "cue cadence should submit bounded work"
    );
    assert!(
        stats.cue_observations > 0,
        "at least one asynchronous estimate should complete"
    );
    assert!(stats.cue_overload_skips <= stats.cue_dispatches);

    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
}

#[test]
fn ready_detector_completion_is_consumed_before_a_due_frame_dispatch() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || acknowledge_and_drain_live_session(listener));
    let (mut detector, calls) = model_and_calls();
    let (release_tx, release_rx) = mpsc::channel();
    detector.first_call_release = Some(release_rx);
    let mut runtime = phase_runtime(
        &request,
        address,
        detector,
        3,
        Duration::from_millis(120),
        phase_test_schedule(10.0, 100.0, Duration::from_millis(800), 3),
    );
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    assert!(matches!(
        runtime.receive_one(Duration::from_secs(2)).unwrap(),
        CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(_)))
    ));
    wait_for_detector_calls(&calls, 1);
    release_tx.send(()).unwrap();

    // The source holds frame two for longer than the configured detector
    // interval, making the first result ready before a full-layout slot is due.
    assert!(matches!(
        runtime.receive_one(Duration::from_secs(2)).unwrap(),
        CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(_)))
    ));
    wait_for_detector_calls(&calls, 2);

    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
}

#[test]
fn exact_request_preflight_v3_readiness_and_rgb8_scene_run_end_to_end() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || receive_live_scene(listener));
    let source_session_id = "gateway-session-73d507";
    let (mut runtime, calls, releases, _) =
        runtime_with_session_id(&request, address, None, Some(source_session_id), 1);
    assert!(matches!(
        runtime.lifecycle(),
        LiveCaptureLifecycle::Pending { .. }
    ));
    let expected_session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready {
            session_id: expected_session_id
        }
    );

    let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) =
        runtime.receive_one(Duration::from_secs(2)).unwrap()
    else {
        panic!("runtime should process and publish the deterministic frame");
    };
    assert_eq!(result.scene.source.session_id, runtime.session_id());
    assert_eq!(result.scene.source.frame_id, "frame-1");
    assert_eq!(result.scene.balls.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let (hello, scene) = receiver.join().unwrap();
    assert_eq!(hello["source_session_id"], source_session_id);
    assert_eq!(
        hello["capture_session_ack"]["source_report"]["session_id"],
        source_session_id
    );
    assert_eq!(
        hello["capture_session_ack"]["request_sha256"],
        runtime.request_sha256()
    );
    assert_eq!(scene["source"]["session_id"], source_session_id);
    assert_eq!(scene["source"]["frame_id"], "frame-1");
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    assert!(matches!(
        runtime.lifecycle(),
        LiveCaptureLifecycle::Stopped { .. }
    ));
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn v2_live_session_acknowledges_output_geometry_and_processes_transformed_frame_size() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_v2_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || receive_live_scene(listener));
    let (mut runtime, calls, releases, _) = runtime(&request, address, None, 1);
    let expected_session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready {
            session_id: expected_session_id
        }
    );

    let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) =
        runtime.receive_one(Duration::from_secs(2)).unwrap()
    else {
        panic!("v2 runtime should process the source-reported output geometry");
    };
    assert_eq!(
        (result.scene.source.width_px, result.scene.source.height_px),
        (12, 12)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let (hello, scene) = receiver.join().unwrap();
    let acknowledgement = &hello["capture_session_ack"];
    assert_eq!(acknowledgement["schema_version"], 2);
    assert_eq!(
        acknowledgement["source_report"]["output_frame_geometry"],
        json!({"width_px": 12, "height_px": 12})
    );
    assert_eq!(scene["source"]["width_px"].as_u64(), Some(12));
    assert_eq!(scene["source"]["height_px"].as_u64(), Some(12));
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn invalid_supervisor_session_identity_is_rejected_before_source_open() {
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let settings = CaptureSettingsSnapshotV1::parse_json(&request).unwrap();
    let opens = Arc::new(AtomicUsize::new(0));
    let (detector, _) = model_and_calls();

    let result = LiveCaptureRuntime::start_with_session_id(
        &request,
        "invalid\nsession-identity",
        detector,
        tracking(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
        ),
        pipeline_config(
            address,
            settings.configuration_revision,
            settings.expected.maximum_frame_age(),
        ),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::clone(&opens),
            frame_count: 1,
            frame_interval: Duration::ZERO,
        },
    );

    assert!(result.is_err());
    assert_eq!(opens.load(Ordering::SeqCst), 0);
}

#[test]
fn rejected_source_preflight_never_exposes_receiver_or_runs_detector() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let settings = CaptureSettingsSnapshotV1::parse_json(&request).unwrap();
    let original = metadata(&settings.expected);
    // Use an explicit requested-profile mismatch, which the source preflight
    // rejects before it can perform any reads or send a scene.
    let format = CapturePixelFormat::new("RGB8").unwrap();
    let wrong_profile =
        CaptureVideoProfile::new(16, 16, CaptureFrameRate::new(60, 1).unwrap(), format).unwrap();
    let mismatched = CaptureSourceMetadata::new(
        original.device_id(),
        original.device_name().map(str::to_owned),
        original.backend_id(),
        wrong_profile,
        original.negotiated_profile().clone(),
        original.capabilities().clone(),
        original.applied_frame_geometry().clone(),
        original.applied_controls().clone(),
        original.timestamp_semantics().clone(),
    )
    .unwrap();
    let (mut runtime, calls, releases, _) = runtime(&request, address, Some(mismatched), 1);
    assert!(matches!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        LiveCaptureLifecycle::Failed { .. }
    ));
    assert!(runtime.receive_one(Duration::ZERO).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(runtime.pipeline_stats().scenes_submitted, 0);
    assert!(
        listener.accept().is_err(),
        "rejected preflight must not start scene transport"
    );
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn mismatched_v3_ready_echo_fails_closed_without_scene_delivery() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let hello = read_message(&mut stream);
        let acknowledgement = &hello["capture_session_ack"];
        write_message(
            &mut stream,
            &json!({
                "type": "ready",
                "protocol_version": 3,
                "configuration_revision": acknowledgement["configuration_revision"],
                "request_sha256": "0".repeat(64),
                "last_transport_sequence": null
            }),
        );
        let mut next = [0_u8; 1];
        let no_scene = matches!(stream.read(&mut next), Ok(0) | Err(_));
        (hello, no_scene)
    });
    let (mut runtime, calls, releases, _) = runtime(&request, address, None, 1);
    assert!(matches!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        LiveCaptureLifecycle::Failed { .. }
    ));
    assert!(runtime.receive_one(Duration::ZERO).is_err());
    let (hello, no_scene) = receiver.join().unwrap();
    assert_eq!(hello["protocol_version"], 3);
    assert!(
        no_scene,
        "a mismatched readiness echo must close before scene data"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(runtime.pipeline_stats().scenes_submitted, 0);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn readiness_timeout_retains_source_ownership_until_explicit_release() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let settings = CaptureSettingsSnapshotV1::parse_json(&request).unwrap();
    let metadata = metadata(&settings.expected);
    let interrupted = Arc::new(AtomicBool::new(false));
    let (started_tx, started_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    let (detector, _) = model_and_calls();
    let mut runtime = LiveCaptureRuntime::start(
        &request,
        detector,
        tracking(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
        ),
        pipeline_config(
            address,
            settings.configuration_revision,
            settings.expected.maximum_frame_age(),
        ),
        BlockingOpenFactory {
            metadata,
            interrupted: Arc::clone(&interrupted),
            started: started_tx,
            resume: resume_rx,
        },
    )
    .unwrap();
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(matches!(
        runtime.wait_until_ready(Duration::from_millis(1)),
        LiveCaptureLifecycle::Pending { .. }
    ));
    assert!(matches!(
        runtime.stop(Duration::from_millis(1)),
        billiards_perception::capture::CaptureRuntimeWait::TimedOut
    ));
    assert!(matches!(
        runtime.owner_status(),
        CaptureOwnerStatus::Owned { .. }
    ));
    assert!(matches!(
        runtime.lifecycle(),
        LiveCaptureLifecycle::Failed { .. }
    ));

    resume_tx.send(()).unwrap();
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    assert_eq!(runtime.owner_status(), CaptureOwnerStatus::Available);
    assert!(matches!(
        runtime.lifecycle(),
        LiveCaptureLifecycle::Stopped { .. }
    ));
}

#[test]
fn duplicate_pipeline_configuration_is_rejected_before_factory_open() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let settings = CaptureSettingsSnapshotV1::parse_json(&request).unwrap();
    let opens = Arc::new(AtomicUsize::new(0));
    let interrupted = Arc::new(AtomicBool::new(false));
    let (detector, _) = model_and_calls();
    let result = LiveCaptureRuntime::start(
        &request,
        detector,
        tracking(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
        ),
        pipeline_config(
            address,
            settings.configuration_revision + 1,
            Duration::from_secs(1),
        ),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted,
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::clone(&opens),
            frame_count: 1,
            frame_interval: Duration::ZERO,
        },
    );
    assert!(result.is_err());
    assert_eq!(opens.load(Ordering::SeqCst), 0);
}
