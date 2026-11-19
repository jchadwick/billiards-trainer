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
use billiards_perception::phase_scheduler::{MotionEvidence, PhaseCadence, SchedulerConfig};
use billiards_perception::table_visibility::{
    derive_reference_from_confirmed_frame, NormalizedPoint, OperatorConfirmedFrame,
    Rgb8Frame as VisibilityFrame, TableRegion, TableVisibilityConfig, TableVisibilityReference,
};
use billiards_perception::tracking::TrackingConfigV1;
use billiards_perception::SceneTransportConfig;
use serde_json::{json, Value};

const SMALL_FRAME_BYTES: usize = 16 * 16 * 3;
const NOT_VISIBLE_FROM_FRAME: usize = 30;
const UNKNOWN_EXPOSURE_FRAME_MARKER: u8 = 60;
const MAXIMUM_MESSAGE_BYTES: usize = 64 * 1024;
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct TestDetector {
    model: ModelIdentity,
    settings: InferenceSettings,
    calls: Arc<AtomicUsize>,
    first_call_release: Option<mpsc::Receiver<()>>,
    blocked_call_release: Option<(usize, mpsc::Receiver<()>)>,
    blocked_frame_marker_releases: Vec<(u8, mpsc::Sender<()>, mpsc::Receiver<()>)>,
    fail_on_frame_marker: Option<u8>,
    shifting_cue_detection: bool,
    wide_cue_detection: bool,
    empty_on_uniform_frame: bool,
    empty_on_call: Option<usize>,
    empty_on_frame_marker: Option<u8>,
    shift_ball_on_uniform_frame: bool,
    include_secondary_ball: bool,
    secondary_ball_center_xy_px: [f32; 2],
    omit_secondary_ball_on_call: Option<usize>,
    omit_secondary_ball_on_frame_markers: Vec<u8>,
}

impl Rgb8FrameDetector for TestDetector {
    fn infer_rgb8(
        &mut self,
        width_px: u32,
        height_px: u32,
        pixels: &[u8],
    ) -> anyhow::Result<FrameInferenceOutput> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        anyhow::ensure!(pixels.len() == (width_px * height_px * 3) as usize);
        if call_index == 0 {
            if let Some(release) = self.first_call_release.as_ref() {
                release.recv().map_err(|error| anyhow::anyhow!(error))?;
            }
        }
        if let Some((blocked_index, release)) = self.blocked_call_release.as_ref() {
            if call_index == *blocked_index {
                release.recv().map_err(|error| anyhow::anyhow!(error))?;
            }
        }
        let frame_marker = pixels[0];
        if let Some((_, started, release)) = self
            .blocked_frame_marker_releases
            .iter()
            .find(|(blocked_marker, _, _)| frame_marker == *blocked_marker)
        {
            started.send(()).map_err(|error| anyhow::anyhow!(error))?;
            release.recv().map_err(|error| anyhow::anyhow!(error))?;
        }
        if self.fail_on_frame_marker == Some(frame_marker) {
            anyhow::bail!("scripted detector failure for frame marker {frame_marker}");
        }
        let move_ball =
            self.shift_ball_on_uniform_frame && pixels.iter().all(|value| *value == 127);
        let (center_x, center_y) = if move_ball { (12.0, 12.0) } else { (6.0, 7.0) };
        let mut detections = vec![Detection {
            class_id: 1,
            class_name: "ball".to_owned(),
            confidence: 0.95,
            bbox_xyxy_px: [
                center_x - 3.0,
                center_y - 3.0,
                center_x + 3.0,
                center_y + 3.0,
            ],
            center_xy_px: [center_x, center_y],
        }];
        if self.wide_cue_detection {
            let (bbox_xyxy_px, center_xy_px) = if width_px > 16 || height_px > 16 {
                ([20.0, 20.0, 180.0, 120.0], [100.0, 70.0])
            } else {
                ([3.0, 3.0, 12.0, 13.0], [7.5, 8.0])
            };
            detections.push(Detection {
                class_id: 0,
                class_name: "cue".to_owned(),
                confidence: 0.9,
                bbox_xyxy_px,
                center_xy_px,
            });
        } else if self.shifting_cue_detection {
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
        if self.include_secondary_ball
            && self.omit_secondary_ball_on_call != Some(call_index)
            && !self
                .omit_secondary_ball_on_frame_markers
                .contains(&frame_marker)
        {
            let [center_x, center_y] = self.secondary_ball_center_xy_px;
            detections.push(Detection {
                class_id: 1,
                class_name: "ball".to_owned(),
                confidence: 0.95,
                bbox_xyxy_px: [
                    center_x - 3.0,
                    center_y - 3.0,
                    center_x + 3.0,
                    center_y + 3.0,
                ],
                center_xy_px: [center_x, center_y],
            });
        }
        if (self.empty_on_uniform_frame && pixels.iter().all(|value| *value == 127))
            || self.empty_on_call == Some(call_index)
            || self.empty_on_frame_marker == Some(frame_marker)
        {
            detections.clear();
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
    motion_on_frame: Option<usize>,
    frame_pixels: Option<fn(usize) -> Vec<u8>>,
    frame_gate: Option<FrameDeliveryGate>,
}

/// Test-only source barrier for proving which frames were admitted before a
/// blocked detector completion. Each listed frame is held until the test
/// releases that exact frame number.
struct FrameDeliveryGate {
    frame_numbers: Vec<usize>,
    reached: mpsc::Sender<usize>,
    release: mpsc::Receiver<usize>,
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
            motion_on_frame: self.motion_on_frame,
            frame_pixels: self.frame_pixels,
            frame_gate: self.frame_gate,
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
    motion_on_frame: Option<usize>,
    frame_pixels: Option<fn(usize) -> Vec<u8>>,
    frame_gate: Option<FrameDeliveryGate>,
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
        if self
            .frame_gate
            .as_ref()
            .is_some_and(|gate| gate.frame_numbers.first() == Some(&self.emitted))
        {
            let gate = self.frame_gate.as_mut().expect("checked above");
            gate.reached
                .send(self.emitted)
                .map_err(|error| error.to_string())?;
            let released = loop {
                match gate.release.recv_timeout(Duration::from_millis(25)) {
                    Ok(released) => break released,
                    Err(mpsc::RecvTimeoutError::Timeout)
                        if self.interrupted.load(Ordering::Acquire) =>
                    {
                        return Ok(CaptureReadOutcome::End);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err("test frame gate was dropped before release".to_owned());
                    }
                }
            };
            if released != self.emitted {
                return Err(format!(
                    "frame {} was released by a gate for frame {released}",
                    self.emitted
                ));
            }
            gate.frame_numbers.remove(0);
            if gate.frame_numbers.is_empty() {
                self.frame_gate = None;
            }
        }
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
        if let Some(render) = self.frame_pixels {
            let rendered = render(self.emitted);
            assert_eq!(rendered.len(), pixels.as_mut_slice().len());
            pixels.as_mut_slice().copy_from_slice(&rendered);
        } else {
            pixels.as_mut_slice().fill(127);
        }
        if self
            .motion_on_frame
            .is_some_and(|frame| self.emitted >= frame)
        {
            let row_stride_bytes = width_px as usize * 3;
            let changed_half_row_bytes = row_stride_bytes / 2;
            for row in 0..height_px as usize {
                let start = row * row_stride_bytes;
                pixels.as_mut_slice()[start..start + changed_half_row_bytes].fill(0);
            }
        }
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
            motion_on_frame: None,
            frame_pixels: None,
            frame_gate: None,
        }))
    }
}

fn request_json() -> String {
    request_json_with_dimensions(16, 16)
}

fn request_json_with_dimensions(width_px: u32, height_px: u32) -> String {
    let mut request: Value = serde_json::from_str(include_str!(
        "../contracts/capture-settings/v1/capture-settings-default.json"
    ))
    .unwrap();
    for profile_name in ["requested_profile", "expected_negotiated_profile"] {
        request["requested"][profile_name]["width_px"] = json!(width_px);
        request["requested"][profile_name]["height_px"] = json!(height_px);
    }
    request["requested"]["maximum_frame_age_ms"] = json!(1_000);
    let frame_bytes = width_px as usize * height_px as usize * 3;
    request["requested"]["maximum_rgb_frame_bytes"] = json!(frame_bytes);
    request["requested"]["maximum_aggregate_rgb_frame_bytes"] = json!(frame_bytes * 2);
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
        blocked_call_release: None,
        blocked_frame_marker_releases: Vec::new(),
        fail_on_frame_marker: None,
        shifting_cue_detection: false,
        wide_cue_detection: false,
        empty_on_uniform_frame: false,
        empty_on_call: None,
        empty_on_frame_marker: None,
        shift_ball_on_uniform_frame: false,
        include_secondary_ball: false,
        secondary_ball_center_xy_px: [11.0, 12.0],
        omit_secondary_ball_on_call: None,
        omit_secondary_ball_on_frame_markers: Vec::new(),
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
        capture_calibration_revision: 0,
        image_space: BatchImageSpace::Raw,
        maximum_frame_age,
        phase_schedule: None,
        localization_profile: None,
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
    tracking_with_gate(camera_profile_id, rectification_revision, 40.0)
}

fn tracking_with_gate(
    camera_profile_id: &str,
    rectification_revision: u64,
    association_gate_px: f64,
) -> TrackingConfigV1 {
    TrackingConfigV1::new(
        1,
        Some(0),
        camera_profile_id,
        association_gate_px,
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
        motion_on_frame: None,
        frame_pixels: None,
        frame_gate: None,
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

fn acknowledge_and_collect_live_session(listener: TcpListener, sender: mpsc::Sender<Value>) {
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
    while let Ok(bytes) = read_framed_message(&mut stream, MAXIMUM_MESSAGE_BYTES) {
        if let Ok(message) = serde_json::from_slice::<Value>(&bytes) {
            if sender.send(message).is_err() {
                break;
            }
        }
    }
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
    let visibility_config = TableVisibilityConfig::default();
    let reference =
        confirmed_reference_from_pixels(&guidance_table_reference_pixels(), &visibility_config);
    LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        request,
        "phase-runtime-visible-test-session",
        detector,
        tracking(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted,
            releases,
            opens,
            frame_count,
            frame_interval,
            motion_on_frame: None,
            frame_pixels: Some(guidance_table_pixels_for_frame),
            frame_gate: None,
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

/// Pause on frame 2 until the low-rate motion worker has had time to settle
/// the repeated visible image. Releasing it makes frame 2 the due motion
/// sample; the later unknown frame can then be delivered between motion
/// checks while ball scans continue at their higher cadence.
fn release_visible_frame_two_at_motion_cadence(
    runtime: &mut LiveCaptureRuntime<TestDetector>,
    frame_gate_reached: &mpsc::Receiver<usize>,
    frame_gate_release: &mpsc::Sender<usize>,
    wire_messages: &mut Vec<Value>,
    delivered: &mpsc::Receiver<Value>,
    deadline: std::time::Instant,
) {
    let mut frame_two_is_held = false;
    while !frame_two_is_held {
        assert!(
            std::time::Instant::now() < deadline,
            "the source should reach its frame 2 barrier"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if let Ok(frame) = frame_gate_reached.try_recv() {
            assert_eq!(frame, 2, "frame 2 must be the visible motion sample");
            frame_two_is_held = true;
        }
        wire_messages.extend(delivered.try_iter());
    }
    while runtime.phase_stats().motion_checks == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the runtime should establish its initial motion baseline"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        wire_messages.extend(delivered.try_iter());
    }

    let motion_due_window = std::time::Instant::now() + Duration::from_millis(1_100);
    while std::time::Instant::now() < motion_due_window {
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        wire_messages.extend(delivered.try_iter());
    }
    frame_gate_release.send(2).unwrap();
}

#[test]
fn cue_detections_do_not_block_clean_full_layout_scans() {
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
        "fresh motion evidence and a clean visible scan should trust the layout: {evidence:?}"
    );
    assert!(runtime.pipeline_stats().scenes_submitted > 0);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
}

#[test]
fn exposure_shifted_visibility_reference_does_not_block_live_scan_publication() {
    check_exposure_shift_publication(false);
}

#[test]
fn covered_table_with_extreme_shift_cannot_trust_an_empty_scan_or_clear_prior_balls() {
    check_exposure_shift_publication(true);
}

#[test]
fn empty_scan_uses_visibility_of_its_own_frame_after_later_frame_becomes_visible() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let bright_reference: Vec<u8> = exposure_table_pixels(0)
        .into_iter()
        .map(|value| value.saturating_add(120))
        .collect();
    let derived = derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(
            VisibilityFrame {
                width: 16,
                height: 16,
                pixels: &bright_reference,
            },
            TableRegion::new([
                NormalizedPoint { x: 0.1, y: 0.1 },
                NormalizedPoint { x: 0.9, y: 0.1 },
                NormalizedPoint { x: 0.9, y: 0.9 },
                NormalizedPoint { x: 0.1, y: 0.9 },
            ]),
        ),
        &visibility_config,
    )
    .unwrap();
    let reference = derived.reference.unwrap();

    let (mut detector, calls) = model_and_calls();
    let (release_tx, release_rx) = mpsc::channel();
    detector.first_call_release = Some(release_rx);
    detector.empty_on_uniform_frame = true;
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(30.0, 20.0, Duration::from_millis(1), 2));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "frame-local-visibility-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 8,
            frame_interval: Duration::from_millis(20),
            motion_on_frame: None,
            frame_pixels: Some(unknown_first_then_bright_table),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    runtime.receive_one(Duration::from_secs(2)).unwrap();
    wait_for_detector_calls(&calls, 1);
    for _ in 0..5 {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    release_tx.send(()).unwrap();

    for _ in 0..5 {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
        if runtime.phase_stats().visibility_rejected_empty_scans > 0 {
            break;
        }
    }
    assert!(
        runtime.phase_stats().visibility_rejected_empty_scans > 0,
        "the delayed empty scan from the unconfirmed frame must be rejected even after a newer frame is visible"
    );
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    assert!(deliveries
        .iter()
        .any(|delivery| { delivery["perception"]["table_visibility"]["status"] == "visible" }));
    assert!(
        !deliveries.iter().any(|delivery| {
            delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(Vec::is_empty)
        }),
        "an empty scan from the earlier unconfirmed frame must not clear published centers"
    );
}

#[test]
fn delayed_visible_empty_scan_stays_blocked_after_unknown_then_visible_frames() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference = confirmed_reference_from_pixels(&exposure_table_pixels(0), &visibility_config);

    let (mut detector, calls) = model_and_calls();
    let (release_tx, release_rx) = mpsc::channel();
    detector.blocked_call_release = Some((4, release_rx));
    detector.empty_on_call = Some(4);
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(30.0, 20.0, Duration::from_millis(1), 1));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "visible-unknown-visible-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 16,
            frame_interval: Duration::from_millis(60),
            motion_on_frame: None,
            frame_pixels: Some(visible_unknown_visible_with_subthreshold_steps),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let blocked_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while calls.load(Ordering::SeqCst) < 5 && std::time::Instant::now() < blocked_deadline {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        5,
        "frame 5 detector call must be in flight"
    );
    let trusted_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !runtime.scheduler_evidence().unwrap().unwrap().map_trusted
        && runtime.phase_stats().captured_frames < 8
        && std::time::Instant::now() < trusted_deadline
    {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    assert!(
        runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "the delayed scan must be in flight while prior visible scans have established trust"
    );
    let mut wire_messages = delivered.try_iter().collect::<Vec<_>>();
    let visible_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while !(wire_messages.iter().any(|delivery| {
        delivery["perception"]["table_visibility"]["status"] == "unknown"
            && delivery["perception"]["table_visibility"]["observation_frame_id"] == "frame-8"
    }) && wire_messages.iter().any(|delivery| {
        delivery["perception"]["table_visibility"]["status"] == "visible"
            && delivery["perception"]["table_visibility"]["observation_frame_id"] == "frame-9"
    })) {
        assert!(
            std::time::Instant::now() < visible_deadline,
            "newer Unknown then Visible evidence must be assessed while frame 5 inference is blocked"
        );
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
        wire_messages.extend(delivered.try_iter());
    }
    let evidence = runtime.scheduler_evidence().unwrap().unwrap();
    assert_eq!(evidence.motion, Some(MotionEvidence::Quiet));
    assert!(!evidence.map_trusted);
    release_tx.send(()).unwrap();
    let obsolete = runtime.receive_one(Duration::from_secs(2)).unwrap();
    assert!(matches!(
        obsolete,
        CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(
            billiards_perception::capture_pipeline::CaptureSkipReason::UnverifiedEmptyLayout
        )))
    ));

    assert!(
        runtime.phase_stats().visibility_rejected_empty_scans > 0,
        "a visible scan captured before newer uncertain coverage must not clear centers"
    );
    assert!(!runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    let recovery_deadline = std::time::Instant::now() + Duration::from_secs(3);
    while !runtime.scheduler_evidence().unwrap().unwrap().map_trusted {
        assert!(
            std::time::Instant::now() < recovery_deadline,
            "a later settled visible scan should be eligible to restore trust"
        );
        runtime.receive_one(Duration::from_secs(2)).unwrap();
    }
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    wire_messages.extend(delivered.try_iter());
    let deliveries = wire_messages;
    assert!(
        deliveries.iter().any(|delivery| {
            delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(|balls| !balls.is_empty())
        }),
        "the earlier real center should remain available after the delayed empty scan is rejected"
    );
    assert!(!deliveries.iter().any(|delivery| {
        delivery["source_snapshot"]["balls"]
            .as_array()
            .is_some_and(Vec::is_empty)
    }));
}

#[test]
fn skipped_empty_not_visible_scan_publishes_invalidation_for_its_own_frame() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference = confirmed_bright_table_reference(&visibility_config);

    let (mut detector, calls) = model_and_calls();
    detector.empty_on_call = Some(1);
    let (release_tx, release_rx) = mpsc::channel();
    detector.blocked_call_release = Some((1, release_rx));
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    // Only frame 1 receives a motion/visibility check. The second detector
    // frame is assessed when its blocked inference completes, exercising the
    // skipped-result path that cannot rely on a motion-triggered phase event.
    config.phase_schedule = Some(phase_test_schedule(30.0, 1.0, Duration::from_millis(1), 1));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "skipped-not-visible-invalidation-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 5,
            frame_interval: Duration::from_millis(60),
            motion_on_frame: None,
            frame_pixels: Some(not_visible_on_frame_two_then_visible),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    runtime.receive_one(Duration::from_secs(2)).unwrap();
    wait_for_detector_calls(&calls, 1);
    runtime.receive_one(Duration::from_secs(2)).unwrap();
    wait_for_detector_calls(&calls, 2);
    runtime.receive_one(Duration::from_secs(2)).unwrap();
    release_tx.send(()).unwrap();
    let completion = runtime.receive_one(Duration::from_secs(2)).unwrap();
    assert!(matches!(
        completion,
        CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(
            billiards_perception::capture_pipeline::CaptureSkipReason::UnverifiedEmptyLayout
        )))
    ));
    assert!(!runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    assert!(
        deliveries.iter().any(|delivery| {
            delivery["source_snapshot"].is_null()
                && delivery["perception"]["decision_frame_id"] == "frame-2"
                && delivery["perception"]["table_visibility"]["status"] == "not_visible"
                && delivery["perception"]["layout_trusted"] == false
                && delivery["perception"]["guidance_valid"] == false
        }),
        "a skipped empty scan must still publish its frame-specific invalidation: {deliveries:?}"
    );
    assert!(deliveries.iter().any(|delivery| {
        delivery["source_snapshot"]["source"]["frame_id"] == "frame-1"
            && delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(|balls| !balls.is_empty())
    }));
    assert!(!deliveries.iter().any(|delivery| {
        delivery["source_snapshot"]["balls"]
            .as_array()
            .is_some_and(Vec::is_empty)
    }));
}

#[test]
fn over_age_not_visible_scan_publishes_invalidation_before_obsolete_skip() {
    let _guard = TEST_LOCK.lock().unwrap();
    let mut request: Value = serde_json::from_str(&request_json()).unwrap();
    request["requested"]["maximum_frame_age_ms"] = json!(100);
    let request = request.to_string();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference = confirmed_bright_table_reference(&visibility_config);

    let (mut detector, calls) = model_and_calls();
    let (release_tx, release_rx) = mpsc::channel();
    detector.blocked_call_release = Some((1, release_rx));
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(30.0, 1.0, Duration::from_millis(1), 1));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "over-age-not-visible-invalidation-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 5,
            frame_interval: Duration::from_millis(60),
            motion_on_frame: None,
            frame_pixels: Some(not_visible_on_frame_two_then_visible),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    runtime.receive_one(Duration::from_secs(2)).unwrap();
    wait_for_detector_calls(&calls, 1);
    runtime.receive_one(Duration::from_secs(2)).unwrap();
    wait_for_detector_calls(&calls, 2);
    runtime.receive_one(Duration::from_secs(2)).unwrap();
    runtime.receive_one(Duration::from_secs(2)).unwrap();
    release_tx.send(()).unwrap();
    let completion = runtime.receive_one(Duration::from_secs(2)).unwrap();
    assert!(matches!(
        completion,
        CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(
            billiards_perception::capture_pipeline::CaptureSkipReason::ObsoleteInference
        )))
    ));
    assert!(!runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    assert!(
        deliveries.iter().any(|delivery| {
            delivery["source_snapshot"].is_null()
                && delivery["perception"]["decision_frame_id"] == "frame-2"
                && delivery["perception"]["table_visibility"]["status"] == "not_visible"
                && delivery["perception"]["layout_trusted"] == false
                && delivery["perception"]["guidance_valid"] == false
        }),
        "an over-age partial result must still publish frame-specific invalidation: {deliveries:?}"
    );
    assert!(
        !deliveries
            .iter()
            .any(|delivery| { delivery["source_snapshot"]["source"]["frame_id"] == "frame-2" }),
        "over-age detections must not publish as a scene"
    );
    assert!(deliveries.iter().any(|delivery| {
        delivery["source_snapshot"]["source"]["frame_id"] == "frame-1"
            && delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(|balls| !balls.is_empty())
    }));
}

#[test]
fn not_visible_partial_scan_invalidates_trust_and_publishes_measured_center() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference = confirmed_bright_table_reference(&visibility_config);
    let (mut detector, _calls) = model_and_calls();
    detector.include_secondary_ball = true;
    // The lower-right measured center is near a pocket in this calibrated
    // fixture. It must remain remembered when coverage becomes NotVisible.
    detector.secondary_ball_center_xy_px = [12.5, 12.5];
    detector.omit_secondary_ball_on_frame_markers = vec![127];
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    // Ball work runs faster than visibility/motion sampling so the detector
    // can observe an unconfirmed frame after trust is established and before
    // the next appearance check.
    config.phase_schedule = Some(phase_test_schedule(30.0, 10.0, Duration::from_millis(1), 1));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "partial-not-visible-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 40,
            frame_interval: Duration::from_millis(30),
            motion_on_frame: None,
            frame_pixels: Some(visible_until_frame_twenty_nine_then_not_visible),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let readiness_deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !runtime.scheduler_evidence().unwrap().unwrap().map_trusted
        && runtime.phase_stats().captured_frames < (NOT_VISIBLE_FROM_FRAME as u64)
    {
        assert!(
            std::time::Instant::now() < readiness_deadline,
            "the visible frames should establish a trusted layout before the table-wide visibility change"
        );
        runtime.receive_one(Duration::from_secs(2)).unwrap();
    }
    assert!(
        runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "a clean scan should establish the initial trusted layout"
    );
    let partial_deadline = std::time::Instant::now() + Duration::from_secs(5);
    let partial_result = loop {
        assert!(
            std::time::Instant::now() < partial_deadline,
            "a measured partial result should arrive from the table-wide NotVisible frames"
        );
        let outcome = runtime.receive_one(Duration::from_secs(2)).unwrap();
        let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) = outcome
        else {
            continue;
        };
        let source_frame_number = result
            .batch
            .source
            .frame_id
            .strip_prefix("frame-")
            .and_then(|number| number.parse::<usize>().ok())
            .unwrap_or_default();
        if source_frame_number >= NOT_VISIBLE_FROM_FRAME {
            break result;
        }
    };
    let result = partial_result;
    let partial_frame_id = result.scene.source.frame_id.clone();
    assert_eq!(result.scene.balls.len(), 2);
    assert!(result.scene.balls.iter().any(|ball| {
        ball.status == billiards_perception::BallStatus::Observed
            && ball.measurement.is_some()
            && ball.last_observation.frame_id == result.scene.source.frame_id
    }));
    let retained_pocket_side_ball = result
        .scene
        .balls
        .iter()
        .find(|ball| ball.last_observation.frame_id != result.scene.source.frame_id)
        .expect("the unmatched pocket-side ball should retain its last-good center");
    assert_eq!(
        retained_pocket_side_ball.status,
        billiards_perception::BallStatus::Occluded
    );
    let retained_center = retained_pocket_side_ball
        .measurement
        .as_ref()
        .unwrap()
        .image_center_px;
    assert_eq!(retained_center.x, 12.5);
    assert_eq!(retained_center.y, 12.5);
    assert!(
        !runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "appearance conflict must invalidate trust before publishing partial detections"
    );
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    let first_wire_scene_for_frame = deliveries
        .iter()
        .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == partial_frame_id)
        .expect("the exact partial detector frame should have a wire scene");
    assert_eq!(
        first_wire_scene_for_frame["perception"]["layout_trusted"],
        false
    );
    assert_eq!(
        first_wire_scene_for_frame["perception"]["guidance_valid"],
        false
    );
    assert!(
        first_wire_scene_for_frame["current_state"]["payload"]["balls"]
            .as_array()
            .is_some_and(|balls| {
                balls.iter().any(|ball| {
                    ball["tracking_status"] == "observed" && ball["position_kind"] == "measured"
                })
            }),
        "the first wire scene for the exact frame must carry its measured center and invalid guidance: {deliveries:?}"
    );
    let wire_balls = first_wire_scene_for_frame["current_state"]["payload"]["balls"]
        .as_array()
        .unwrap();
    let retained_wire_ball = wire_balls
        .iter()
        .find(|ball| ball["id"].as_str() == Some(retained_pocket_side_ball.track_id.as_str()))
        .expect("the exact partial wire scene must keep the pocket-side last-good center");
    assert_eq!(
        retained_wire_ball["source_observation"]["frame_id"],
        retained_pocket_side_ball.last_observation.frame_id
    );
    let retained_source_ball = first_wire_scene_for_frame["source_snapshot"]["balls"]
        .as_array()
        .unwrap()
        .iter()
        .find(|ball| ball["track_id"] == retained_pocket_side_ball.track_id)
        .expect("the wire source scene must preserve the retained track record");
    assert_eq!(
        retained_source_ball["measurement"]["image_center_px"]["x"],
        retained_center.x
    );
    assert_eq!(
        retained_source_ball["measurement"]["image_center_px"]["y"],
        retained_center.y
    );
}

#[test]
fn unknown_nonempty_scan_updates_measurements_without_dropping_unmatched_centers() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference = confirmed_reference_from_pixels(&exposure_table_pixels(0), &visibility_config);

    let (mut detector, calls) = model_and_calls();
    detector.include_secondary_ball = true;
    detector.omit_secondary_ball_on_frame_markers = vec![60];
    let (release_tx, release_rx) = mpsc::channel();
    let (detector_started_tx, detector_started_rx) = mpsc::channel();
    detector.blocked_frame_marker_releases = vec![(60, detector_started_tx, release_rx)];
    let (frame_gate_reached_tx, frame_gate_reached_rx) = mpsc::channel();
    let (frame_gate_release_tx, frame_gate_release_rx) = mpsc::channel();
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    let mut schedule = phase_test_schedule(60.0, 7.0, Duration::from_millis(1), 1);
    schedule.motion_evidence_expiry = Duration::from_secs(10);
    schedule.layout_evidence_expiry = Duration::from_secs(10);
    config.phase_schedule = Some(schedule);
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "unknown-nonempty-partial-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 28,
            frame_interval: Duration::from_millis(20),
            motion_on_frame: None,
            frame_pixels: Some(visible_until_frame_nineteen_then_unknown),
            frame_gate: Some(FrameDeliveryGate {
                frame_numbers: vec![19, 20],
                reached: frame_gate_reached_tx,
                release: frame_gate_release_rx,
            }),
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let detector_deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut original_track_ids = None;
    let mut frame_nineteen_is_held = false;
    while !frame_nineteen_is_held
        || original_track_ids.is_none()
        || !runtime.scheduler_evidence().unwrap().unwrap().map_trusted
    {
        assert!(
            std::time::Instant::now() < detector_deadline,
            "visible scans should establish a trusted map while frame 19 is held"
        );
        if let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) =
            runtime.receive_one(Duration::from_millis(20)).unwrap()
        {
            if result.scene.balls.len() == 2 {
                original_track_ids.get_or_insert_with(|| {
                    result
                        .scene
                        .balls
                        .iter()
                        .map(|ball| ball.track_id.clone())
                        .collect::<Vec<_>>()
                });
            }
        }
        frame_nineteen_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
            assert_eq!(frame, 19);
            true
        });
    }
    assert!(runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    let original_track_ids =
        original_track_ids.expect("visible scans should publish two tracked balls");
    let motion_checks_before_unknown = runtime.phase_stats().motion_checks;
    frame_gate_release_tx.send(19).unwrap();
    let mut unknown_detector_started = false;
    let mut frame_twenty_is_held = false;
    while !unknown_detector_started || !frame_twenty_is_held {
        assert!(
            std::time::Instant::now() < detector_deadline,
            "the detector should block on exact Unknown frame 19 while frame 20 remains held"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        unknown_detector_started |= detector_started_rx.try_recv().is_ok();
        frame_twenty_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
            assert_eq!(frame, 20);
            true
        });
    }
    assert!(calls.load(Ordering::SeqCst) >= 19);
    assert_eq!(
        runtime.phase_stats().motion_checks,
        motion_checks_before_unknown
    );
    release_tx.send(()).unwrap();
    let unknown_partial = loop {
        assert!(
            std::time::Instant::now() < detector_deadline,
            "the exact Unknown detector frame should complete while frame 20 remains held"
        );
        let outcome = runtime.receive_one(Duration::from_millis(20)).unwrap();
        if let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) = outcome
        {
            if result.scene.source.frame_id == "frame-19" {
                break result;
            }
        }
    };
    assert_eq!(unknown_partial.scene.source.frame_id, "frame-19");
    assert_eq!(unknown_partial.scene.balls.len(), 2);
    assert!(unknown_partial.scene.balls.iter().any(|ball| {
        ball.status == billiards_perception::BallStatus::Observed && ball.measurement.is_some()
    }));
    for prior_track_id in &original_track_ids {
        assert!(
            unknown_partial.scene.balls.iter().any(|ball| {
                &ball.track_id == prior_track_id
                    && (ball.status == billiards_perception::BallStatus::Observed
                        || ball.last_observation.frame_id != "frame-19")
            }),
            "partial Unknown coverage must preserve each last-good track id and its original observation when unmatched: {:?}",
            unknown_partial.scene.balls
        );
    }
    assert!(
        !runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "Unknown coverage on a detector frame between motion checks must revoke layout trust"
    );
    let mut wire_messages = Vec::new();
    let exact_frame_scene = loop {
        wire_messages.extend(delivered.try_iter());
        if let Some(delivery) = wire_messages
            .iter()
            .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-19")
        {
            break delivery.clone();
        }
        assert!(
            std::time::Instant::now() < detector_deadline,
            "the exact Unknown partial scene should reach the wire while frame 20 remains held"
        );
        assert!(matches!(
            runtime.receive_one(Duration::from_millis(20)).unwrap(),
            CaptureReceiveOutcome::TimedOut
        ));
    };
    assert_eq!(exact_frame_scene["perception"]["layout_trusted"], false);
    assert_eq!(exact_frame_scene["perception"]["guidance_valid"], false);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    wire_messages.extend(delivered.try_iter());
    assert_eq!(
        exact_frame_scene["perception"]["table_visibility"]["status"],
        "unknown"
    );
    assert_eq!(exact_frame_scene["perception"]["layout_trusted"], false);
    assert_eq!(exact_frame_scene["perception"]["guidance_valid"], false);
    let wire_balls = exact_frame_scene["current_state"]["payload"]["balls"]
        .as_array()
        .unwrap();
    assert_eq!(wire_balls.len(), 2);
    assert!(wire_balls.iter().any(|ball| {
        ball["tracking_status"] == "observed" && ball["position_kind"] == "measured"
    }));
    assert!(wire_balls.iter().any(|ball| {
        ball["tracking_status"] != "observed" && ball["position_kind"] == "last_observed"
    }));
}

#[test]
fn unknown_partial_first_scene_revokes_previously_valid_guidance() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference =
        confirmed_reference_from_pixels(&guidance_table_reference_pixels(), &visibility_config);

    let (mut detector, _calls) = model_and_calls();
    detector.wide_cue_detection = true;
    detector.include_secondary_ball = true;
    detector.omit_secondary_ball_on_frame_markers = vec![UNKNOWN_EXPOSURE_FRAME_MARKER];
    let (detector_started_tx, detector_started_rx) = mpsc::channel();
    let (detector_release_tx, detector_release_rx) = mpsc::channel();
    detector.blocked_frame_marker_releases = vec![(
        UNKNOWN_EXPOSURE_FRAME_MARKER,
        detector_started_tx,
        detector_release_rx,
    )];
    let (frame_gate_reached_tx, frame_gate_reached_rx) = mpsc::channel();
    let (frame_gate_release_tx, frame_gate_release_rx) = mpsc::channel();
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    let mut schedule = phase_test_schedule(60.0, 1.0, Duration::from_millis(1), 1);
    schedule.recovering.cue_hz = 60.0;
    schedule.pre_shot.cue_hz = 60.0;
    schedule.aiming.cue_hz = 60.0;
    schedule.motion_evidence_expiry = Duration::from_secs(10);
    schedule.cue_evidence_expiry = Duration::from_secs(10);
    schedule.layout_evidence_expiry = Duration::from_secs(10);
    config.phase_schedule = Some(schedule);
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "unknown-partial-guidance-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 10,
            frame_interval: Duration::from_millis(80),
            motion_on_frame: None,
            frame_pixels: Some(guidance_visible_until_frame_five_then_unknown),
            frame_gate: Some(FrameDeliveryGate {
                frame_numbers: vec![2, 5, 6],
                reached: frame_gate_reached_tx,
                release: frame_gate_release_rx,
            }),
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut wire_messages = Vec::new();
    release_visible_frame_two_at_motion_cadence(
        &mut runtime,
        &frame_gate_reached_rx,
        &frame_gate_release_tx,
        &mut wire_messages,
        &delivered,
        deadline,
    );
    let mut unknown_frame_is_held = false;
    while !unknown_frame_is_held
        || !wire_messages.iter().any(|delivery: &Value| {
            delivery["source_snapshot"].is_object()
                && delivery["perception"]["guidance_valid"] == true
                && delivery["perception"]["layout_trusted"] == true
                && delivery["source_snapshot"]["source"]["frame_id"] != "frame-5"
        })
    {
        assert!(
            std::time::Instant::now() < deadline,
            "a visible frame should publish valid guidance before the partial frame"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) => {}
            CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !unknown_frame_is_held {
            unknown_frame_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
                assert_eq!(frame, 5, "the source must stop before the Unknown frame");
                true
            });
        }
        wire_messages.extend(delivered.try_iter());
    }
    assert!(runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    assert!(
        runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );
    let motion_checks_before_unknown = runtime.phase_stats().motion_checks;
    frame_gate_release_tx.send(5).unwrap();
    let mut source_frame_six_is_held = false;
    while detector_started_rx.try_recv().is_err() {
        assert!(
            std::time::Instant::now() < deadline,
            "the detector should reach the source-marked Unknown frame"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !source_frame_six_is_held {
            source_frame_six_is_held = frame_gate_reached_rx
                .try_recv()
                .is_ok_and(|frame| frame == 6);
        }
        wire_messages.extend(delivered.try_iter());
    }
    if !source_frame_six_is_held {
        assert_eq!(
            frame_gate_reached_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            6,
            "the next appearance frame must remain held while the detector is blocked"
        );
        source_frame_six_is_held = true;
    }
    assert!(
        source_frame_six_is_held,
        "the next appearance frame must remain held while the detector is blocked"
    );
    assert_eq!(
        runtime.phase_stats().motion_checks,
        motion_checks_before_unknown
    );
    assert!(runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    assert!(
        runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );

    detector_release_tx.send(()).unwrap();
    let unknown_partial = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the blocked detector result should complete after release"
        );
        let outcome = runtime.receive_one(Duration::from_millis(20)).unwrap();
        wire_messages.extend(delivered.try_iter());
        if let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) = outcome
        {
            if result.scene.source.frame_id == "frame-5" {
                break result;
            }
        }
    };
    assert_eq!(unknown_partial.scene.balls.len(), 2);
    assert!(unknown_partial.scene.balls.iter().any(|ball| {
        ball.status == billiards_perception::BallStatus::Observed && ball.measurement.is_some()
    }));
    assert!(!runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    assert!(
        !runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );
    while !wire_messages
        .iter()
        .any(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-5")
    {
        wire_messages.extend(delivered.try_iter());
        assert!(
            std::time::Instant::now() < deadline,
            "the exact partial wire scene should arrive before frame 6 is released"
        );
        assert!(matches!(
            runtime.receive_one(Duration::from_millis(20)).unwrap(),
            CaptureReceiveOutcome::TimedOut
        ));
    }
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    wire_messages.extend(delivered.try_iter());

    let frame_scenes: Vec<_> = wire_messages
        .iter()
        .filter(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-5")
        .collect();
    let first_wire_scene = frame_scenes
        .first()
        .expect("the exact partial detector frame should publish a wire scene");
    assert_eq!(frame_scenes.len(), 1);
    assert_eq!(first_wire_scene["perception"]["layout_trusted"], false);
    assert_eq!(first_wire_scene["perception"]["guidance_valid"], false);
    assert!(first_wire_scene["current_state"]["payload"]["balls"]
        .as_array()
        .is_some_and(|balls| {
            balls.iter().any(|ball| {
                ball["tracking_status"] == "observed" && ball["position_kind"] == "measured"
            }) && balls.iter().any(|ball| {
                ball["tracking_status"] != "observed" && ball["position_kind"] == "last_observed"
            })
        }));
}

#[test]
fn failed_detector_on_unknown_frame_publishes_invalidation_before_returning() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference =
        confirmed_reference_from_pixels(&guidance_table_reference_pixels(), &visibility_config);

    let (mut detector, _calls) = model_and_calls();
    detector.wide_cue_detection = true;
    detector.include_secondary_ball = true;
    detector.fail_on_frame_marker = Some(UNKNOWN_EXPOSURE_FRAME_MARKER);
    let (detector_started_tx, detector_started_rx) = mpsc::channel();
    let (detector_release_tx, detector_release_rx) = mpsc::channel();
    detector.blocked_frame_marker_releases = vec![(
        UNKNOWN_EXPOSURE_FRAME_MARKER,
        detector_started_tx,
        detector_release_rx,
    )];
    let (frame_gate_reached_tx, frame_gate_reached_rx) = mpsc::channel();
    let (frame_gate_release_tx, frame_gate_release_rx) = mpsc::channel();
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    let mut schedule = phase_test_schedule(60.0, 1.0, Duration::from_millis(1), 1);
    schedule.recovering.cue_hz = 60.0;
    schedule.pre_shot.cue_hz = 60.0;
    schedule.aiming.cue_hz = 60.0;
    schedule.motion_evidence_expiry = Duration::from_secs(10);
    schedule.cue_evidence_expiry = Duration::from_secs(10);
    schedule.layout_evidence_expiry = Duration::from_secs(10);
    config.phase_schedule = Some(schedule);
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "failed-unknown-frame-invalidation-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 10,
            frame_interval: Duration::from_millis(80),
            motion_on_frame: None,
            frame_pixels: Some(guidance_visible_until_frame_five_then_unknown),
            frame_gate: Some(FrameDeliveryGate {
                frame_numbers: vec![2, 5, 6],
                reached: frame_gate_reached_tx,
                release: frame_gate_release_rx,
            }),
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut wire_messages = Vec::new();
    release_visible_frame_two_at_motion_cadence(
        &mut runtime,
        &frame_gate_reached_rx,
        &frame_gate_release_tx,
        &mut wire_messages,
        &delivered,
        deadline,
    );
    let mut unknown_frame_is_held = false;
    while !unknown_frame_is_held
        || !wire_messages.iter().any(|delivery: &Value| {
            delivery["source_snapshot"].is_object()
                && delivery["perception"]["layout_trusted"] == true
                && delivery["perception"]["guidance_valid"] == true
        })
    {
        if std::time::Instant::now() >= deadline {
            wire_messages.extend(delivered.try_iter());
            panic!(
                "visible frames should establish valid guidance before the failed detector frame; evidence={:?}, stats={:?}, calls={}, deliveries={wire_messages:?}",
                runtime.scheduler_evidence().unwrap(),
                runtime.phase_stats(),
                _calls.load(Ordering::SeqCst)
            );
        }
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !unknown_frame_is_held {
            unknown_frame_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
                assert_eq!(frame, 5);
                true
            });
        }
        wire_messages.extend(delivered.try_iter());
    }
    let evidence_before_failure = runtime.scheduler_evidence().unwrap().unwrap();
    assert!(evidence_before_failure.map_trusted && evidence_before_failure.guidance_valid);
    let motion_checks_before_failure = runtime.phase_stats().motion_checks;
    frame_gate_release_tx.send(5).unwrap();

    let mut frame_six_is_held = false;
    while detector_started_rx.try_recv().is_err() {
        if std::time::Instant::now() >= deadline {
            panic!(
                "the detector should be blocked on the source-marked frame 5; calls={}, stats={:?}, evidence={:?}",
                _calls.load(Ordering::SeqCst),
                runtime.phase_stats(),
                runtime.scheduler_evidence().unwrap()
            );
        }
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !frame_six_is_held {
            frame_six_is_held = frame_gate_reached_rx
                .try_recv()
                .is_ok_and(|frame| frame == 6);
        }
        wire_messages.extend(delivered.try_iter());
    }
    if !frame_six_is_held {
        assert_eq!(
            frame_gate_reached_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            6
        );
    }
    assert_eq!(
        runtime.phase_stats().motion_checks,
        motion_checks_before_failure
    );
    assert!(
        runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );

    detector_release_tx.send(()).unwrap();
    frame_gate_release_tx.send(6).unwrap();
    loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the failed detector result should be consumed"
        );
        match runtime.receive_one(Duration::from_secs(2)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(
                billiards_perception::capture_pipeline::CaptureSkipReason::DetectorFailed,
            ))) => break,
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
    }
    assert_eq!(runtime.phase_stats().detector_failures, 1);
    assert!(!runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    assert!(
        !runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    wire_messages.extend(delivered.try_iter());

    assert!(
        wire_messages.iter().any(|delivery| {
            delivery["source_snapshot"].is_null()
                && delivery["perception"]["decision_frame_id"] == "frame-5"
                && delivery["perception"]["table_visibility"]["status"] == "unknown"
                && delivery["perception"]["layout_trusted"] == false
                && delivery["perception"]["guidance_valid"] == false
        }),
        "a failed detector call must still publish the exact frame's visibility invalidation: {wire_messages:?}"
    );
}

#[test]
fn missing_appearance_reference_keeps_nonempty_scans_partial_and_rejects_empty_clear() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let (mut detector, calls) = model_and_calls();
    detector.include_secondary_ball = true;
    detector.omit_secondary_ball_on_call = Some(1);
    detector.empty_on_call = Some(2);
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(30.0, 20.0, Duration::from_millis(1), 1));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "missing-optional-appearance-reference-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        None,
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 8,
            frame_interval: Duration::from_millis(60),
            motion_on_frame: None,
            frame_pixels: Some(stable_bright_exposure_table_pixels),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while runtime.phase_stats().visibility_rejected_empty_scans == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the scan without a visibility reference should remain partial, and the later empty scan should be rejected"
        );
        runtime.receive_one(Duration::from_secs(2)).unwrap();
    }
    assert!(calls.load(Ordering::SeqCst) >= 3);
    assert!(
        !runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "missing positive table coverage must not establish whole-layout trust"
    );
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    assert!(
        deliveries.iter().any(|delivery| {
            delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(|balls| !balls.is_empty())
        }),
        "nonempty detector measurements should still publish: {deliveries:?}"
    );
    assert!(deliveries.iter().any(|delivery| {
        delivery["perception"]["table_visibility"]["reason"] == "missing_reference"
    }));
    let omitted_ball_frame = deliveries
        .iter()
        .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-2")
        .expect("the one-ball partial scan should still publish");
    assert_eq!(
        omitted_ball_frame["source_snapshot"]["balls"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(omitted_ball_frame["perception"]["layout_trusted"], false);
    assert!(
        !deliveries.iter().any(|delivery| {
            delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(Vec::is_empty)
        }),
        "an empty result without positive visibility evidence must never clear the remembered layout"
    );
}

#[test]
fn phase_aware_empty_scan_without_visibility_monitor_cannot_clear_last_good_centers() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let (mut detector, calls) = model_and_calls();
    detector.include_secondary_ball = true;
    detector.omit_secondary_ball_on_call = Some(1);
    detector.empty_on_call = Some(2);
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(30.0, 20.0, Duration::from_millis(1), 1));
    config.scene_transport.public_state_delivery_enabled = true;
    // This compatibility constructor intentionally supplies no visibility
    // monitor. Phase-aware empty replacement must still fail closed without
    // explicit positive table coverage evidence.
    let mut runtime = LiveCaptureRuntime::start_with_session_id_and_audit(
        &request,
        "empty-scan-without-visibility-monitor-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 8,
            frame_interval: Duration::from_millis(60),
            motion_on_frame: None,
            frame_pixels: Some(stable_bright_exposure_table_pixels),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while runtime.phase_stats().visibility_rejected_empty_scans == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the nonempty scans without a monitor should remain partial and the empty scan should be rejected"
        );
        runtime.receive_one(Duration::from_millis(100)).unwrap();
    }
    assert!(calls.load(Ordering::SeqCst) >= 3);
    assert!(
        !runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "a nonempty scan without positive coverage must remain partial"
    );
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    assert!(
        deliveries.iter().any(|delivery| {
            delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(|balls| !balls.is_empty())
        }),
        "the previous measured centers must remain published: {deliveries:?}"
    );
    let omitted_ball_frame = deliveries
        .iter()
        .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-2")
        .expect("the one-ball partial scan should still publish");
    assert_eq!(
        omitted_ball_frame["source_snapshot"]["balls"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(omitted_ball_frame["perception"]["layout_trusted"], false);
    assert!(
        !deliveries.iter().any(|delivery| {
            delivery["source_snapshot"]["balls"]
                .as_array()
                .is_some_and(Vec::is_empty)
        }),
        "an empty scan with no visibility monitor must not publish a clear layout"
    );
}

#[test]
fn live_wire_preserves_partial_centers_then_reconciles_pocket_and_empty_scans() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    let reference =
        confirmed_reference_from_pixels(&guidance_table_reference_pixels(), &visibility_config);

    let (mut detector, _calls) = model_and_calls();
    detector.wide_cue_detection = true;
    detector.include_secondary_ball = true;
    detector.omit_secondary_ball_on_frame_markers = vec![60, 61, 62, 49, 63];
    detector.empty_on_frame_marker = Some(50);
    let (partial_started_tx, partial_started_rx) = mpsc::channel();
    let (partial_release_tx, partial_release_rx) = mpsc::channel();
    let (reconcile_started_tx, reconcile_started_rx) = mpsc::channel();
    let (reconcile_release_tx, reconcile_release_rx) = mpsc::channel();
    let (empty_started_tx, empty_started_rx) = mpsc::channel();
    let (empty_release_tx, empty_release_rx) = mpsc::channel();
    detector.blocked_frame_marker_releases = vec![
        (60, partial_started_tx, partial_release_rx),
        (49, reconcile_started_tx, reconcile_release_rx),
        (50, empty_started_tx, empty_release_rx),
    ];
    let (frame_gate_reached_tx, frame_gate_reached_rx) = mpsc::channel();
    let (frame_gate_release_tx, frame_gate_release_rx) = mpsc::channel();
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    let mut schedule = phase_test_schedule(30.0, 1.0, Duration::from_millis(1), 1);
    schedule.recovering.cue_hz = 30.0;
    schedule.pre_shot.cue_hz = 30.0;
    schedule.aiming.cue_hz = 30.0;
    schedule.motion_evidence_expiry = Duration::from_secs(10);
    schedule.cue_evidence_expiry = Duration::from_secs(10);
    schedule.layout_evidence_expiry = Duration::from_secs(10);
    config.phase_schedule = Some(schedule);
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "partial-to-clean-live-wire-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            40.0,
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 11,
            frame_interval: Duration::from_millis(80),
            motion_on_frame: None,
            frame_pixels: Some(visible_unknown_and_reconciliation_frames),
            frame_gate: Some(FrameDeliveryGate {
                frame_numbers: vec![2, 5, 6, 8, 9, 10, 11],
                reached: frame_gate_reached_tx,
                release: frame_gate_release_rx,
            }),
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(8);
    let mut wire_messages = Vec::new();
    release_visible_frame_two_at_motion_cadence(
        &mut runtime,
        &frame_gate_reached_rx,
        &frame_gate_release_tx,
        &mut wire_messages,
        &delivered,
        deadline,
    );
    let mut frame_five_is_held = false;
    while !frame_five_is_held
        || !wire_messages.iter().any(|delivery: &Value| {
            delivery["source_snapshot"].is_object()
                && delivery["perception"]["layout_trusted"] == true
                && delivery["perception"]["guidance_valid"] == true
                && delivery["source_snapshot"]["source"]["frame_id"] != "frame-5"
        })
    {
        if std::time::Instant::now() >= deadline {
            wire_messages.extend(delivered.try_iter());
            panic!(
                "visible full scans should establish trusted guidance before the uncertain frame; held={}, evidence={:?}, stats={:?}, scenes={:?}",
                frame_five_is_held,
                runtime.scheduler_evidence().unwrap(),
                runtime.phase_stats(),
                wire_messages.iter()
                    .filter(|delivery| delivery["source_snapshot"].is_object())
                    .map(|delivery| (
                        delivery["source_snapshot"]["source"]["frame_id"].clone(),
                        delivery["perception"]["layout_trusted"].clone(),
                        delivery["perception"]["guidance_valid"].clone(),
                    ))
                    .collect::<Vec<_>>()
            );
        }
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !frame_five_is_held {
            frame_five_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
                assert_eq!(frame, 5);
                true
            });
        }
        wire_messages.extend(delivered.try_iter());
    }
    assert!(
        runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );
    frame_gate_release_tx.send(5).unwrap();

    let mut frame_six_is_held = false;
    while partial_started_rx.try_recv().is_err() {
        if std::time::Instant::now() >= deadline {
            panic!(
                "the frame 5 detector should block after capture admission; calls={}, stats={:?}, evidence={:?}, frame6_held={}",
                _calls.load(Ordering::SeqCst),
                runtime.phase_stats(),
                runtime.scheduler_evidence().unwrap(),
                frame_six_is_held
            );
        }
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !frame_six_is_held {
            frame_six_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
                assert_eq!(frame, 6);
                true
            });
        }
        wire_messages.extend(delivered.try_iter());
    }
    if !frame_six_is_held {
        assert_eq!(
            frame_gate_reached_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            6
        );
    }
    partial_release_tx.send(()).unwrap();

    let partial = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "the uncertain scan should publish measured and retained centers"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result)))
                if result.scene.source.frame_id == "frame-5" =>
            {
                break result
            }
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        wire_messages.extend(delivered.try_iter());
    };
    let first_partial_wire_scene = loop {
        wire_messages.extend(delivered.try_iter());
        if let Some(delivery) = wire_messages
            .iter()
            .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-5")
        {
            break delivery.clone();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the exact partial scene should reach the wire before the next frame is released"
        );
        assert!(matches!(
            runtime.receive_one(Duration::from_millis(20)).unwrap(),
            CaptureReceiveOutcome::TimedOut
        ));
    };
    assert_eq!(partial.scene.balls.len(), 2);
    assert!(partial.scene.balls.iter().any(|ball| {
        ball.status == billiards_perception::BallStatus::Observed && ball.measurement.is_some()
    }));
    assert!(
        partial.scene.balls.iter().any(|ball| {
            ball.status != billiards_perception::BallStatus::Observed
                && ball.last_observation.frame_id != "frame-5"
        }),
        "the unmatched pre-existing ball should retain its prior observation: {:?}",
        partial.scene.balls
    );
    assert!(
        !runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );
    assert_eq!(
        first_partial_wire_scene["source_snapshot"]["balls"]
            .as_array()
            .unwrap()
            .len(),
        2,
        "both the current measurement and unmatched last-good center must be present"
    );
    assert_eq!(
        first_partial_wire_scene["perception"]["layout_trusted"],
        false
    );
    assert_eq!(
        first_partial_wire_scene["perception"]["guidance_valid"],
        false
    );
    frame_gate_release_tx.send(6).unwrap();

    let mut frame_eight_is_held = false;
    while !frame_eight_is_held
        || !runtime
            .scheduler_evidence()
            .unwrap()
            .is_some_and(|evidence| evidence.map_trusted && evidence.guidance_valid)
    {
        assert!(
            std::time::Instant::now() < deadline,
            "settled visible scans should recover guidance before the clean reconciliation"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !frame_eight_is_held {
            frame_eight_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
                assert_eq!(frame, 8);
                true
            });
        }
        wire_messages.extend(delivered.try_iter());
    }
    frame_gate_release_tx.send(8).unwrap();

    let mut frame_nine_is_held = false;
    while reconcile_started_rx.try_recv().is_err() {
        assert!(
            std::time::Instant::now() < deadline,
            "the clean visible scan should enter the detector"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !frame_nine_is_held {
            frame_nine_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
                assert_eq!(frame, 9);
                true
            });
        }
        wire_messages.extend(delivered.try_iter());
    }
    if !frame_nine_is_held {
        assert_eq!(
            frame_gate_reached_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            9
        );
    }
    reconcile_release_tx.send(()).unwrap();
    frame_gate_release_tx.send(9).unwrap();

    let reconciled = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "a positively covered clean scan should remove the absent track"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result)))
                if result.scene.source.frame_id == "frame-8" =>
            {
                break result
            }
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        wire_messages.extend(delivered.try_iter());
    };
    assert_eq!(reconciled.scene.balls.len(), 1);
    assert!(
        runtime
            .scheduler_evidence()
            .unwrap()
            .unwrap()
            .guidance_valid
    );

    let mut frame_ten_is_held = false;
    while !frame_ten_is_held {
        assert!(
            std::time::Instant::now() < deadline,
            "the source should reach the verified-empty frame"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        frame_ten_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
            assert_eq!(frame, 10);
            true
        });
        wire_messages.extend(delivered.try_iter());
    }
    assert!(runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    frame_gate_release_tx.send(10).unwrap();

    let mut frame_eleven_is_held = false;
    while empty_started_rx.try_recv().is_err() {
        assert!(
            std::time::Instant::now() < deadline,
            "the verified-empty detector call should reach its frame barrier"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !frame_eleven_is_held {
            frame_eleven_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
                assert_eq!(frame, 11);
                true
            });
        }
        wire_messages.extend(delivered.try_iter());
    }
    if !frame_eleven_is_held {
        assert_eq!(
            frame_gate_reached_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            11
        );
    }
    empty_release_tx.send(()).unwrap();
    frame_gate_release_tx.send(11).unwrap();

    let verified_empty = loop {
        assert!(
            std::time::Instant::now() < deadline,
            "a positively visible empty scan should verify the empty layout"
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result)))
                if result.scene.source.frame_id == "frame-10" =>
            {
                break result
            }
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        wire_messages.extend(delivered.try_iter());
    };
    assert!(verified_empty.scene.balls.is_empty());
    let empty_evidence = runtime.scheduler_evidence().unwrap().unwrap();
    assert!(empty_evidence.map_trusted && empty_evidence.empty_layout_verified);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    wire_messages.extend(delivered.try_iter());

    let first_partial_wire = wire_messages
        .iter()
        .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-5")
        .expect("the partial frame should be present on the first-wire lifecycle");
    assert_eq!(first_partial_wire["perception"]["guidance_valid"], false);
    assert!(first_partial_wire["current_state"]["payload"]["balls"]
        .as_array()
        .is_some_and(|balls| {
            balls.len() == 2
                && balls.iter().any(|ball| {
                    ball["tracking_status"] == "observed" && ball["position_kind"] == "measured"
                })
                && balls.iter().any(|ball| {
                    ball["tracking_status"] != "observed"
                        && ball["position_kind"] == "last_observed"
                })
        }));

    let clean_wire = wire_messages
        .iter()
        .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-8")
        .expect("the clean scan should reconcile the absent ball on the wire");
    assert_eq!(
        clean_wire["perception"]["table_visibility"]["status"],
        "visible"
    );
    assert_eq!(clean_wire["perception"]["layout_trusted"], true);
    assert_eq!(clean_wire["perception"]["guidance_valid"], true);
    assert_eq!(
        clean_wire["source_snapshot"]["balls"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    let empty_wire = wire_messages
        .iter()
        .find(|delivery| delivery["source_snapshot"]["source"]["frame_id"] == "frame-10")
        .expect("the verified empty scan should be present on the wire");
    assert_eq!(
        empty_wire["perception"]["table_visibility"]["status"],
        "visible"
    );
    assert_eq!(empty_wire["perception"]["layout_trusted"], true);
    assert_eq!(empty_wire["perception"]["empty_layout_verified"], false);
    assert!(empty_wire["source_snapshot"]["balls"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(empty_wire["current_state"]["payload"]["balls"]
        .as_array()
        .unwrap()
        .is_empty());
    let verified_empty_event = wire_messages
        .iter()
        .find(|delivery| {
            delivery["source_snapshot"].is_null()
                && delivery["perception"]["decision_frame_id"] == "frame-10"
                && delivery["perception"]["empty_layout_verified"] == true
        })
        .expect("the post-scan status event should confirm the verified empty layout");
    assert_eq!(verified_empty_event["perception"]["layout_trusted"], true);
    assert!(verified_empty_event["current_state"]["payload"]["balls"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn nonempty_scan_from_unconfirmed_view_publishes_observation_without_trust() {
    check_exposure_shift_publication_mode(true, true);
}

fn exposure_table_pixels(_frame: usize) -> Vec<u8> {
    let mut pixels = vec![0; SMALL_FRAME_BYTES];
    for y in 2..14 {
        for x in 2..14 {
            pixels[(y * 16 + x) * 3..(y * 16 + x) * 3 + 3].copy_from_slice(&[80, 120, 80]);
        }
    }
    pixels
}

fn covered_after_acquisition(frame: usize) -> Vec<u8> {
    if frame >= 8 {
        vec![127; SMALL_FRAME_BYTES]
    } else {
        exposure_table_pixels(frame)
    }
}

fn bright_exposure_table_pixels() -> Vec<u8> {
    table_pixels_with_brightness_offset(120)
}

fn table_pixels_with_brightness_offset(offset: u8) -> Vec<u8> {
    exposure_table_pixels(0)
        .into_iter()
        .map(|value| value.saturating_add(offset))
        .collect()
}

fn confirmed_bright_table_reference(
    visibility_config: &TableVisibilityConfig,
) -> TableVisibilityReference {
    confirmed_reference_from_pixels(&bright_exposure_table_pixels(), visibility_config)
}

fn confirmed_reference_from_pixels(
    reference_pixels: &[u8],
    visibility_config: &TableVisibilityConfig,
) -> TableVisibilityReference {
    confirmed_reference_from_pixels_with_geometry(reference_pixels, 16, 16, visibility_config)
}

fn confirmed_reference_from_pixels_with_geometry(
    reference_pixels: &[u8],
    width_px: u32,
    height_px: u32,
    visibility_config: &TableVisibilityConfig,
) -> TableVisibilityReference {
    let derived = derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(
            VisibilityFrame {
                width: width_px,
                height: height_px,
                pixels: reference_pixels,
            },
            TableRegion::new([
                NormalizedPoint { x: 0.1, y: 0.1 },
                NormalizedPoint { x: 0.9, y: 0.1 },
                NormalizedPoint { x: 0.9, y: 0.9 },
                NormalizedPoint { x: 0.1, y: 0.9 },
            ]),
        ),
        visibility_config,
    )
    .unwrap();
    assert!(derived.validity.valid, "{:?}", derived.validity.reasons);
    derived.reference.unwrap()
}

fn visible_unknown_visible_with_subthreshold_steps(frame: usize) -> Vec<u8> {
    // The motion monitor is already initialized at +48. Moving to +60 makes
    // the appearance comparison inconclusive (its limit is +52), while the
    // 12-level frame-to-frame step stays below motion's pixel threshold. The
    // return to +48 is also subthreshold, so this tests appearance high-water
    // behavior without relying on a motion-generation change.
    table_pixels_with_brightness_offset(if frame == 8 { 60 } else { 48 })
}

fn visible_until_frame_nineteen_then_unknown(frame: usize) -> Vec<u8> {
    table_pixels_with_brightness_offset(if (19..=20).contains(&frame) { 60 } else { 48 })
}

fn guidance_visible_until_frame_five_then_unknown(frame: usize) -> Vec<u8> {
    render_guidance_table_pixels(if (5..=6).contains(&frame) { 60 } else { 48 })
}

fn visible_unknown_and_reconciliation_frames(frame: usize) -> Vec<u8> {
    let mut pixels = render_guidance_table_pixels(if frame == 5 { 60 } else { 48 });
    // Pixel (0, 0) is outside the calibrated table region and carries only
    // test-script markers. Unique markers keep the pocket-side ball absent
    // from every detector frame between the partial scan and clean removal.
    pixels[0] = match frame {
        5 => 60,
        6 => 61,
        7 => 62,
        8 => 49,
        9 => 63,
        10 => 50,
        _ => 48,
    };
    pixels
}

fn guidance_table_reference_pixels() -> Vec<u8> {
    render_guidance_table_pixels(0)
}

fn guidance_table_pixels_for_frame(_frame: usize) -> Vec<u8> {
    let mut pixels = guidance_table_reference_pixels();
    pixels[0] = 48;
    pixels
}

fn render_guidance_table_pixels(brightness_offset: u8) -> Vec<u8> {
    let width = 16;
    let height = 16;
    let mut pixels = vec![0; SMALL_FRAME_BYTES];
    for y in 2..14 {
        for x in 2..14 {
            let offset = (y * width + x) * 3;
            pixels[offset..offset + 3].copy_from_slice(&[30, 100, 30]);
        }
    }
    draw_test_line_rgb8(
        &mut pixels,
        width,
        height,
        [3.0, 12.0],
        [11.0, 4.0],
        1.2,
        [235, 235, 235],
    );
    pixels
        .into_iter()
        .map(|value| value.saturating_add(brightness_offset))
        .collect()
}

fn draw_test_line_rgb8(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    from: [f64; 2],
    to: [f64; 2],
    half_width: f64,
    color: [u8; 3],
) {
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let length_squared = dx * dx + dy * dy;
    for y in 0..height {
        for x in 0..width {
            let t = (((x as f64 - from[0]) * dx + (y as f64 - from[1]) * dy) / length_squared)
                .clamp(0.0, 1.0);
            let nearest = [from[0] + t * dx, from[1] + t * dy];
            if (x as f64 - nearest[0]).hypot(y as f64 - nearest[1]) <= half_width {
                let offset = (y * width + x) * 3;
                pixels[offset..offset + 3].copy_from_slice(&color);
            }
        }
    }
}

fn stable_bright_exposure_table_pixels(_frame: usize) -> Vec<u8> {
    bright_exposure_table_pixels()
}

fn not_visible_on_frame_two_then_visible(frame: usize) -> Vec<u8> {
    if frame == 2 {
        vec![127; SMALL_FRAME_BYTES]
    } else {
        bright_exposure_table_pixels()
    }
}

fn visible_until_frame_twenty_nine_then_not_visible(frame: usize) -> Vec<u8> {
    if frame >= NOT_VISIBLE_FROM_FRAME {
        vec![127; SMALL_FRAME_BYTES]
    } else {
        bright_exposure_table_pixels()
    }
}

fn unknown_first_then_bright_table(frame: usize) -> Vec<u8> {
    if frame == 1 {
        vec![127; SMALL_FRAME_BYTES]
    } else {
        exposure_table_pixels(frame)
            .into_iter()
            .map(|value| value.saturating_add(120))
            .collect()
    }
}

fn check_exposure_shift_publication(covered: bool) {
    check_exposure_shift_publication_mode(covered, false);
}

fn check_exposure_shift_publication_mode(covered: bool, nonempty_when_covered: bool) {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = request_json();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let settings = parse_capture_settings_json(&request).unwrap();
    let visibility_config = TableVisibilityConfig::default();
    // Preserve real rail structure while shifting every reference channel by
    // 120, reproducing changed illumination without making a blank scene valid.
    let source = exposure_table_pixels(0);
    let bright_reference: Vec<u8> = source.iter().map(|value| value + 120).collect();
    let derived = derive_reference_from_confirmed_frame(
        OperatorConfirmedFrame::new(
            VisibilityFrame {
                width: 16,
                height: 16,
                pixels: &bright_reference,
            },
            TableRegion::new([
                NormalizedPoint { x: 0.1, y: 0.1 },
                NormalizedPoint { x: 0.9, y: 0.1 },
                NormalizedPoint { x: 0.9, y: 0.9 },
                NormalizedPoint { x: 0.1, y: 0.9 },
            ]),
        ),
        &visibility_config,
    )
    .unwrap();
    assert!(derived.validity.valid, "{:?}", derived.validity.reasons);
    let reference = derived.reference.unwrap();
    let (mut detector, calls) = model_and_calls();
    detector.empty_on_uniform_frame = covered && !nonempty_when_covered;
    detector.shift_ball_on_uniform_frame = nonempty_when_covered;
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(30.0, 20.0, Duration::from_millis(1), 2));
    config.scene_transport.public_state_delivery_enabled = true;
    let interrupted = Arc::new(AtomicBool::new(false));
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "exposure-shift-live-session",
        detector,
        tracking_with_gate(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
            if nonempty_when_covered { 5.0 } else { 40.0 },
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted,
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 16,
            frame_interval: Duration::from_millis(60),
            motion_on_frame: None,
            frame_pixels: Some(if covered {
                covered_after_acquisition
            } else {
                exposure_table_pixels
            }),
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    for _ in 0..16 {
        match runtime.receive_one(Duration::from_secs(2)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) => {}
            CaptureReceiveOutcome::TimedOut => {
                panic!("scripted source did not emit its next frame")
            }
            CaptureReceiveOutcome::Closed => break,
            other => panic!("unexpected live receive result: {other:?}"),
        }
        if !covered
            && runtime
                .scheduler_evidence()
                .unwrap()
                .is_some_and(|evidence| evidence.map_trusted)
        {
            break;
        }
    }
    wait_for_detector_calls(&calls, 1);
    assert!(
        !runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "an Unknown coverage assessment must never authorize a full-layout replacement"
    );
    if covered && !nonempty_when_covered {
        assert!(
            runtime.phase_stats().visibility_rejected_empty_scans > 0,
            "empty detector results from the covered/unconfirmed view must be rejected"
        );
    }
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    assert!(
        deliveries.iter().any(|delivery| {
            delivery["source_snapshot"].is_object()
                && delivery["perception"]["table_visibility"]["status"] == "unknown"
                && delivery["current_state"]["payload"]["balls"]
                    .as_array()
                    .is_some_and(|balls| {
                        balls.iter().any(|ball| {
                            ball["tracking_status"] == "observed"
                                && ball["position_kind"] == "measured"
                        })
                    })
        }),
        "Unknown coverage must still publish real detected centers as partial observations: {deliveries:?}"
    );
    assert!(deliveries.iter().any(|delivery| {
        delivery["perception"]["table_visibility"]["status"] == "unknown"
            && delivery["perception"]["table_visibility"]["reason"] == "extreme_brightness_shift"
    }));
    if covered {
        if !nonempty_when_covered {
            let empty_unknown_scenes: Vec<_> = deliveries
                .iter()
                .filter(|delivery| {
                    delivery["perception"]["table_visibility"]["status"] == "unknown"
                        && delivery["perception"]["table_visibility"]["reason"]
                            == "extreme_brightness_shift"
                        && delivery["source_snapshot"]["balls"]
                            .as_array()
                            .is_some_and(Vec::is_empty)
                })
                .collect();
            assert!(
                empty_unknown_scenes.is_empty(),
                "an explicitly unknown view plus an empty detector result must not publish an empty layout"
            );
        }
        let blocked: Vec<_> = deliveries
            .iter()
            .filter(|delivery| {
                delivery["perception"]["table_visibility"]["reason"] == "lost_rail_contrast"
            })
            .collect();
        assert!(!blocked.is_empty());
        assert!(
            blocked.iter().all(|delivery| {
                delivery["perception"]["layout_trusted"] == false
                    && delivery["perception"]["empty_layout_verified"] != true
                    && delivery["current_state"]["payload"]["balls"]
                        .as_array()
                        .is_some_and(|balls| !balls.is_empty())
            }),
            "covered table must retain stale balls, never a trusted empty replacement"
        );
        if nonempty_when_covered {
            assert!(
                blocked.iter().any(|delivery| {
                    delivery["current_state"]["payload"]["balls"]
                        .as_array()
                        .is_some_and(|balls| {
                            !balls.is_empty()
                                && balls.iter().any(|ball| {
                                    ball["tracking_status"] == "observed"
                                        && ball["position_kind"] == "measured"
                                })
                                && balls.iter().any(|ball| {
                                    ball["tracking_status"] == "occluded"
                                        && ball["position_kind"] == "last_observed"
                                })
                        })
                }),
                "an uncertain non-empty scan should publish current detected centers while keeping the whole-table map untrusted: {deliveries:?}"
            );
        }
    }
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
fn slow_clean_worker_result_beyond_capture_deadline_reaches_live_state() {
    let _guard = TEST_LOCK.lock().unwrap();
    let mut request: Value = serde_json::from_str(&request_json()).unwrap();
    request["requested"]["maximum_frame_age_ms"] = json!(100);
    let request = request.to_string();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (delivery_tx, delivery_rx) = mpsc::channel();
    let receiver =
        thread::spawn(move || acknowledge_and_collect_live_session(listener, delivery_tx));
    let (mut detector, calls) = model_and_calls();
    let (release_tx, release_rx) = mpsc::channel();
    detector.first_call_release = Some(release_rx);
    let (slow_release_tx, slow_release_rx) = mpsc::channel();
    detector.blocked_call_release = Some((1, slow_release_rx));
    let mut runtime = phase_runtime(
        &request,
        address,
        detector,
        32,
        Duration::from_millis(100),
        phase_test_schedule(10.0, 20.0, Duration::from_millis(800), 3),
    );
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    // Keep servicing motion evidence while the initial scan is blocked. It
    // was captured before settlement and must not become the trusted layout.
    for _ in 0..11 {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    wait_for_detector_calls(&calls, 1);
    release_tx.send(()).unwrap();
    let first_completion = runtime.receive_one(Duration::from_secs(2)).unwrap();
    assert!(matches!(
        first_completion,
        CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(_)))
    ));
    wait_for_detector_calls(&calls, 2);

    // Let a second scan, captured after quiet settlement, remain blocked for
    // more than the configured 100 ms age limit while fresh motion checks run.
    for _ in 0..4 {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    slow_release_tx.send(()).unwrap();
    let mut published_scene = None;
    for _ in 0..5 {
        if let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) =
            runtime.receive_one(Duration::from_secs(2)).unwrap()
        {
            published_scene = Some(result.scene);
            break;
        }
    }
    let scene = published_scene.expect("slow settled scan should reach the live pipeline");
    assert_eq!(scene.balls.len(), 1);
    assert!(runtime
        .scheduler_evidence()
        .unwrap()
        .is_some_and(|evidence| evidence.map_trusted));
    assert_eq!(runtime.pipeline_stats().scenes_submitted, 1);

    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let published_layout_ages = delivery_rx
        .try_iter()
        .filter_map(|message| {
            message
                .pointer("/perception/last_ball_observation_age_ms")
                .and_then(Value::as_u64)
        })
        .collect::<Vec<_>>();
    assert!(
        published_layout_ages.iter().any(|age| *age >= 300),
        "published state must report inference and completion-slot delay from capture: {published_layout_ages:?}"
    );
}

#[test]
fn motion_during_blocked_scan_publishes_partial_result_without_retrust() {
    let _guard = TEST_LOCK.lock().unwrap();
    let mut request: Value = serde_json::from_str(&request_json()).unwrap();
    request["requested"]["maximum_frame_age_ms"] = json!(1_000);
    let request = request.to_string();
    let settings = parse_capture_settings_json(&request).unwrap();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || acknowledge_and_drain_live_session(listener));
    let (mut detector, calls) = model_and_calls();
    let (release_tx, release_rx) = mpsc::channel();
    detector.blocked_call_release = Some((1, release_rx));
    let interrupted = Arc::new(AtomicBool::new(false));
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(
        10.0,
        20.0,
        Duration::from_millis(800),
        3,
    ));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start(
        &request,
        detector,
        tracking(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
        ),
        config,
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted,
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 20,
            frame_interval: Duration::from_millis(100),
            // Keep the table still long enough for the blocked scan to start
            // even when the target machine coalesces a source frame while the
            // coordinator is draining an earlier detector completion.
            motion_on_frame: Some(8),
            frame_pixels: None,
            frame_gate: None,
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let detector_start_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while calls.load(Ordering::SeqCst) < 2 && std::time::Instant::now() < detector_start_deadline {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    assert!(
        calls.load(Ordering::SeqCst) >= 2,
        "the second inference must be blocked before the scripted movement starts"
    );
    let before_motion = runtime.scheduler_evidence().unwrap().unwrap();
    let motion_deadline = std::time::Instant::now() + Duration::from_secs(2);
    while runtime
        .scheduler_evidence()
        .unwrap()
        .unwrap()
        .motion_generation
        == before_motion.motion_generation
        && std::time::Instant::now() < motion_deadline
    {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    let after_motion = runtime.scheduler_evidence().unwrap().unwrap();
    assert!(after_motion.motion_generation > before_motion.motion_generation);
    assert!(!after_motion.map_trusted);

    release_tx.send(()).unwrap();
    let late = runtime.receive_one(Duration::from_secs(2)).unwrap();
    let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) = late else {
        panic!("a fresh nonempty scan should publish as an untrusted partial observation");
    };
    assert_eq!(result.scene.balls.len(), 1);
    assert_eq!(
        result.scene.balls[0].status,
        billiards_perception::BallStatus::Observed
    );
    assert!(result.scene.balls[0].measurement.is_some());
    assert!(
        !runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "a result from the previous motion generation must not restore layout trust"
    );
    for _ in 0..3 {
        assert!(matches!(
            runtime.receive_one(Duration::from_secs(2)).unwrap(),
            CaptureReceiveOutcome::Frame(Ok(_))
        ));
    }
    let after_quiet = runtime.scheduler_evidence().unwrap().unwrap();
    assert!(
        !after_quiet.map_trusted,
        "quiet continuity requires a new clean scan"
    );
    assert!(runtime.pipeline_stats().scenes_submitted >= 1);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
}

#[test]
fn shot_onset_frame_is_still_detected_but_cannot_replace_layout() {
    let _guard = TEST_LOCK.lock().unwrap();
    let mut request: Value = serde_json::from_str(&request_json()).unwrap();
    request["requested"]["maximum_frame_age_ms"] = json!(1_000);
    let request = request.to_string();
    let settings = parse_capture_settings_json(&request).unwrap();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, delivered) = mpsc::channel();
    let receiver = thread::spawn(move || acknowledge_and_collect_live_session(listener, sender));
    let visibility_config = TableVisibilityConfig::default();
    let reference =
        confirmed_reference_from_pixels(&guidance_table_reference_pixels(), &visibility_config);
    let (mut detector, calls) = model_and_calls();
    let (detector_started_tx, detector_started_rx) = mpsc::channel();
    let (detector_release_tx, detector_release_rx) = mpsc::channel();
    detector.blocked_frame_marker_releases = vec![(0, detector_started_tx, detector_release_rx)];
    let (frame_gate_reached_tx, frame_gate_reached_rx) = mpsc::channel();
    let (frame_gate_release_tx, frame_gate_release_rx) = mpsc::channel();
    let mut config = pipeline_config(
        address,
        settings.configuration_revision,
        settings.expected.maximum_frame_age(),
    );
    config.phase_schedule = Some(phase_test_schedule(30.0, 5.0, Duration::from_millis(1), 1));
    config.scene_transport.public_state_delivery_enabled = true;
    let mut runtime = LiveCaptureRuntime::start_with_session_id_audit_and_visibility(
        &request,
        "shot-onset-frame-detection-session",
        detector,
        tracking(
            settings.expected.camera_profile_id(),
            settings.expected.rectification_revision(),
        ),
        config,
        None,
        visibility_config,
        Some(reference),
        TestFactory {
            metadata: metadata(&settings.expected),
            interrupted: Arc::new(AtomicBool::new(false)),
            releases: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            frame_count: 10,
            frame_interval: Duration::from_millis(100),
            motion_on_frame: Some(5),
            frame_pixels: Some(guidance_table_pixels_for_frame),
            frame_gate: Some(FrameDeliveryGate {
                frame_numbers: vec![5, 6],
                reached: frame_gate_reached_tx,
                release: frame_gate_release_rx,
            }),
        },
    )
    .unwrap();
    let session_id = runtime.session_id().to_owned();
    assert_eq!(
        runtime.wait_until_ready(Duration::from_secs(2)),
        &LiveCaptureLifecycle::Ready { session_id }
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut frame_five_is_held = false;
    while !frame_five_is_held || !runtime.scheduler_evidence().unwrap().unwrap().map_trusted {
        assert!(
            std::time::Instant::now() < deadline,
            "visible scans should establish trust before the shot-onset frame is released; held={}, calls={}, stats={:?}, evidence={:?}",
            frame_five_is_held,
            calls.load(Ordering::SeqCst),
            runtime.phase_stats(),
            runtime.scheduler_evidence().unwrap()
        );
        match runtime.receive_one(Duration::from_millis(20)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(_)) | CaptureReceiveOutcome::TimedOut => {}
            other => panic!("unexpected live receive result: {other:?}"),
        }
        frame_five_is_held = frame_gate_reached_rx.try_recv().is_ok_and(|frame| {
            assert_eq!(frame, 5);
            true
        });
    }
    assert!(runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    let calls_before_onset_frame = calls.load(Ordering::SeqCst);
    let generation_before_onset = runtime
        .scheduler_evidence()
        .unwrap()
        .unwrap()
        .motion_generation;
    frame_gate_release_tx.send(5).unwrap();
    assert!(matches!(
        runtime.receive_one(Duration::from_secs(2)).unwrap(),
        CaptureReceiveOutcome::Frame(Ok(_))
    ));
    detector_started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("the detector must receive the exact frame that establishes shot motion");
    assert!(calls.load(Ordering::SeqCst) > calls_before_onset_frame);
    let generation_after_onset = runtime
        .scheduler_evidence()
        .unwrap()
        .unwrap()
        .motion_generation;
    assert!(generation_after_onset > generation_before_onset);
    assert!(
        !runtime.scheduler_evidence().unwrap().unwrap().map_trusted,
        "motion on the detector frame must prevent whole-layout replacement"
    );

    assert_eq!(
        frame_gate_reached_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap(),
        6,
        "the next source frame stays held while frame five's inference completes"
    );
    detector_release_tx.send(()).unwrap();
    frame_gate_release_tx.send(6).unwrap();
    let mut last_onset_outcome = "no outcome".to_owned();
    let result = loop {
        if std::time::Instant::now() >= deadline {
            panic!(
                "the onset frame should publish its measured ball as a partial result; last={last_onset_outcome}, calls={}, stats={:?}, evidence={:?}",
                calls.load(Ordering::SeqCst),
                runtime.phase_stats(),
                runtime.scheduler_evidence().unwrap()
            );
        }
        match runtime.receive_one(Duration::from_secs(2)).unwrap() {
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result)))
                if result.scene.source.frame_id == "frame-5" =>
            {
                break result;
            }
            CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Skipped(reason))) => {
                last_onset_outcome = format!("skipped: {reason:?}");
            }
            CaptureReceiveOutcome::Frame(Ok(outcome)) => {
                last_onset_outcome = format!("{outcome:?}");
            }
            CaptureReceiveOutcome::TimedOut => last_onset_outcome = "timed out".to_owned(),
            other => panic!("unexpected live receive result: {other:?}"),
        }
    };
    assert_eq!(result.scene.balls.len(), 1);
    assert_eq!(
        result.scene.balls[0].status,
        billiards_perception::BallStatus::Observed
    );
    assert!(result.scene.balls[0].measurement.is_some());
    assert!(!runtime.scheduler_evidence().unwrap().unwrap().map_trusted);
    assert!(matches!(
        runtime.stop(Duration::from_secs(2)),
        billiards_perception::capture::CaptureRuntimeWait::Completed(_)
    ));
    receiver.join().unwrap();
    let deliveries = delivered.try_iter().collect::<Vec<_>>();
    assert!(deliveries.iter().any(|delivery| {
        delivery["source_snapshot"]["source"]["frame_id"] == "frame-5"
            && delivery["perception"]["layout_trusted"] == false
    }));
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
            motion_on_frame: None,
            frame_pixels: None,
            frame_gate: None,
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
            motion_on_frame: None,
            frame_pixels: None,
            frame_gate: None,
        },
    );
    assert!(result.is_err());
    assert_eq!(opens.load(Ordering::SeqCst), 0);
}
