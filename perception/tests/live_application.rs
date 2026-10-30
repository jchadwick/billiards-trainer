//! Process-level live composition test, isolated from ONNX Runtime's
//! process-global environment used by the library unit-test binary.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use billiards_perception::capture::{
    CaptureAppliedCameraControls, CaptureAppliedFrameGeometry, CaptureCapability,
    CaptureControlReadback, CaptureExposureMode, CaptureFrameBudget, CaptureFrameLimit,
    CaptureInterrupt, CaptureOpenError, CaptureReadOutcome, CaptureSource,
    CaptureSourceCapabilities, CaptureSourceFactory, CaptureSourceMetadata,
    CaptureTimestampSemantics, RgbFrameEnvelope,
};
use billiards_perception::capture_settings_wire::CaptureSettingsSnapshotV1;
use billiards_perception::detection_batch::SourceFrame;
use billiards_perception::framing::{read_framed_message, write_framed_message};
use billiards_perception::inference::ProviderSelection;
use billiards_perception::live_application::{
    parse_phase_schedule_json, run_live_application, LiveApplicationSettings,
};
use billiards_perception::SceneTransportConfig;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MAXIMUM_SCENE_BYTES: usize = 64 * 1024;
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn real_detector_and_injected_source_complete_v3_startup_scene_delivery_and_release() {
    let _guard = TEST_LOCK.lock().unwrap();
    let request = test_request_json();
    let resolved = CaptureSettingsSnapshotV1::parse_json(&request).unwrap();
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || receive_one_scene(listener));

    let scratch = tempfile::tempdir().unwrap();
    let onnx_path = scratch.path().join("tiny-two-class.onnx");
    std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tiny_two_class_detector.onnx"
        ),
        &onnx_path,
    )
    .unwrap();
    let checkpoint_path = scratch.path().join("source.pt");
    std::fs::write(&checkpoint_path, b"test checkpoint lineage").unwrap();
    std::fs::write(
        format!("{}.manifest.json", onnx_path.display()),
        serde_json::to_vec(&json!({
            "manifest_schema_version": 1,
            "source_model": { "sha256": sha256_file(&checkpoint_path) },
            "exported_model": {
                "path": "tiny-two-class.onnx",
                "sha256": sha256_file(&onnx_path)
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

    let interrupted = Arc::new(AtomicBool::new(false));
    let releases = Arc::new(AtomicUsize::new(0));
    let factory = TestFactory {
        metadata: test_source_metadata(&resolved.expected),
        interrupted: Arc::clone(&interrupted),
        releases: Arc::clone(&releases),
    };
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let worker_shutdown = Arc::clone(&shutdown_requested);
    let settings = live_settings(onnx_path, checkpoint_path, address, &resolved.expected);
    let worker_request = request.clone();
    let worker = thread::spawn(move || {
        run_live_application(&worker_request, settings, factory, &worker_shutdown)
    });

    let receiver_result = receiver.join();
    shutdown_requested.store(true, Ordering::SeqCst);
    let worker_result = worker.join().unwrap();
    worker_result.expect("live application stopped cleanly");
    let receiver_result = receiver_result.expect("live delivery receiver stopped cleanly");
    let (hello, scene) = receiver_result.expect("runtime connected and delivered one scene");

    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["protocol_version"], 3);
    assert_eq!(
        hello["capture_session_ack"]["source_report"]["session_id"],
        "live-session-test"
    );
    assert_eq!(scene["delivery_schema_version"], 2);
    assert_eq!(scene["engine_instance_id"], "live-engine-test");
    assert_eq!(
        scene["source_snapshot"]["source"]["session_id"],
        "live-session-test"
    );
    assert_eq!(scene["source_snapshot"]["source"]["width_px"], 32);
    assert_eq!(scene["source_snapshot"]["source"]["height_px"], 32);
    assert_eq!(
        scene["source_snapshot"]["balls"].as_array().unwrap().len(),
        1
    );
    assert!(scene["perception"].is_object());
    assert!(
        scene["current_state"]["localization_age_ns"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        scene["current_state"]["ball_age_ns"]["track-1"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(releases.load(Ordering::SeqCst), 1);
    assert!(interrupted.load(Ordering::SeqCst));
}

fn test_request_json() -> String {
    let mut request: Value = serde_json::from_str(include_str!(
        "../contracts/capture-settings/v1/capture-settings-default.json"
    ))
    .unwrap();
    for profile_name in ["requested_profile", "expected_negotiated_profile"] {
        request["requested"][profile_name]["width_px"] = json!(32);
        request["requested"][profile_name]["height_px"] = json!(32);
    }
    request["requested"]["maximum_frame_age_ms"] = json!(1_000);
    request["requested"]["maximum_rgb_frame_bytes"] = json!(32 * 32 * 3);
    request["requested"]["maximum_aggregate_rgb_frame_bytes"] = json!(1_048_576);
    serde_json::to_string(&request).unwrap()
}

fn live_settings(
    onnx_path: std::path::PathBuf,
    source_checkpoint_path: std::path::PathBuf,
    address: SocketAddr,
    expected: &billiards_perception::live_config::ExpectedCaptureSettings,
) -> LiveApplicationSettings {
    LiveApplicationSettings {
        onnx_path: Some(onnx_path),
        source_checkpoint_path,
        provider: ProviderSelection::Cpu,
        image_size: 32,
        confidence: 0.4,
        iou: 0.45,
        max_detections: 300,
        intra_op_threads: Some(1),
        warmup_runs: 1,
        source_image_space: expected.image_space(),
        engine_instance_id: "live-engine-test".to_owned(),
        source_session_id: "live-session-test".to_owned(),
        tracking_ball_class_id: 1,
        tracking_cue_class_id: Some(0),
        tracking_camera_profile_id: expected.camera_profile_id().to_owned(),
        tracking_association_gate_px: 40.0,
        tracking_association_prediction_horizon: Duration::from_millis(100),
        tracking_lost_after: Duration::from_secs(1),
        tracking_maximum_track_age: Duration::from_secs(3),
        tracking_maximum_active_tracks: 300,
        tracking_recent_observation_capacity: 8,
        tracking_configuration_revision: expected.configuration_revision(),
        tracking_rectification_revision: expected.rectification_revision(),
        phase_schedule: parse_phase_schedule_json(
            r#"{"recovering":{"motion_hz":15,"ball_hz":5,"cue_hz":5},"pre_shot":{"motion_hz":15,"ball_hz":1,"cue_hz":5},"aiming":{"motion_hz":15,"ball_hz":1,"cue_hz":5},"shot":{"motion_hz":15,"ball_hz":0,"cue_hz":0},"quiet_dwell_ms":800,"layout_consensus_count":3,"layout_position_tolerance_px":4.0,"motion_evidence_expiry_ms":200,"cue_evidence_expiry_ms":250,"layout_evidence_expiry_ms":1500}"#,
        )
        .unwrap(),
        table_visibility: billiards_perception::table_visibility::TableVisibilityConfig::default(),
        ball_localization: billiards_perception::ball_localization_profile::BallLocalizationSettings::default(),
        scene_transport: SceneTransportConfig {
            address,
            maximum_message_bytes: MAXIMUM_SCENE_BYTES,
            public_state_delivery_enabled: true,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
            reconnect_initial_delay: Duration::from_millis(10),
            reconnect_maximum_delay: Duration::from_millis(20),
        },
        frame_preview: None,
        startup_timeout: Duration::from_secs(5),
        poll_interval: Duration::from_millis(10),
        graceful_stop_timeout: Duration::from_secs(2),
        capture_settings_maximum_bytes: 16 * 1024,
    }
}

fn test_source_metadata(
    expected: &billiards_perception::live_config::ExpectedCaptureSettings,
) -> CaptureSourceMetadata {
    let requested_profile = expected.requested_profile().clone();
    let negotiated_profile = expected.expected_negotiated_profile().clone();
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
    CaptureSourceMetadata::new(
        expected.device_id(),
        Some("deterministic live application source".to_owned()),
        "test-source",
        requested_profile,
        negotiated_profile,
        capabilities,
        CaptureAppliedFrameGeometry::new(
            expected.configuration_revision(),
            expected.camera_profile_id(),
            expected.image_space(),
            expected.rectification_revision(),
        )
        .unwrap(),
        controls,
        CaptureTimestampSemantics::ReadCompletionOnly,
    )
    .unwrap()
}

fn receive_one_scene(listener: TcpListener) -> Result<(Value, Value), String> {
    listener.set_nonblocking(true).unwrap();
    let accept_deadline = Instant::now() + Duration::from_secs(3);
    let (mut stream, _) = loop {
        match listener.accept() {
            Ok(connection) => break connection,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= accept_deadline {
                    return Err("timed out waiting for live process connection".to_owned());
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(error.to_string()),
        }
    };
    stream.set_nonblocking(false).unwrap();
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
    let scene_deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let event = read_message(&mut stream);
        if event["source_snapshot"].is_object() {
            return Ok((hello, event));
        }
        if Instant::now() >= scene_deadline {
            return Err("timed out waiting for a raw scene snapshot".to_owned());
        }
    }
}

fn read_message(stream: &mut TcpStream) -> Value {
    serde_json::from_slice(&read_framed_message(stream, MAXIMUM_SCENE_BYTES).unwrap()).unwrap()
}

fn write_message(stream: &mut TcpStream, message: &Value) {
    write_framed_message(
        stream,
        &serde_json::to_vec(message).unwrap(),
        MAXIMUM_SCENE_BYTES,
    )
    .unwrap();
}

fn sha256_file(path: &std::path::Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

struct TestInterrupt(Arc<AtomicBool>);

impl CaptureInterrupt for TestInterrupt {
    fn interrupt(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct TestFactory {
    metadata: CaptureSourceMetadata,
    interrupted: Arc<AtomicBool>,
    releases: Arc<AtomicUsize>,
}

impl CaptureSourceFactory for TestFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::clone(&self.interrupted)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        Ok(Box::new(TestSource {
            metadata: self.metadata,
            interrupted: self.interrupted,
            releases: self.releases,
            emitted: false,
        }))
    }
}

struct TestSource {
    metadata: CaptureSourceMetadata,
    interrupted: Arc<AtomicBool>,
    releases: Arc<AtomicUsize>,
    emitted: bool,
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
        if !self.emitted {
            self.emitted = true;
            let profile = self.metadata.negotiated_profile();
            let mut pixels = frame_budget
                .allocate_rgb8_buffer(profile.width_px(), profile.height_px(), frame_limit)
                .map_err(|error| error.to_string())?;
            pixels.as_mut_slice().fill(127);
            let frame = RgbFrameEnvelope::new(
                SourceFrame {
                    session_id: session_id.to_owned(),
                    frame_id: "live-frame-1".to_owned(),
                    timestamp: None,
                    width_px: profile.width_px(),
                    height_px: profile.height_px(),
                },
                pixels,
                Instant::now(),
                frame_limit,
            )
            .map_err(|error| error.to_string())?;
            return Ok(CaptureReadOutcome::Frame(frame));
        }

        while !self.interrupted.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(1));
        }
        Ok(CaptureReadOutcome::End)
    }

    fn release(&mut self) -> Result<(), String> {
        self.releases.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
