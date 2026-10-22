//! Hardware-independent integration coverage for the live capture seam.
//!
//! A deterministic source exercises the public capture worker and the real
//! capture-to-scene pipeline. The loopback peer implements only the anonymous
//! protocol-v2 handshake and captures one complete scene snapshot.

use std::collections::BTreeSet;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use billiards_perception::calibration::TableCalibrationV1;
use billiards_perception::calibration::ValidatedTableCalibrationV1;
use billiards_perception::capture::{
    CaptureAggregateFrameLimit, CaptureAppliedCameraControls, CaptureAppliedFrameGeometry,
    CaptureCapability, CaptureFrameBudget, CaptureFrameLimit, CaptureFrameRate, CaptureInterrupt,
    CaptureOpenError, CapturePixelFormat, CapturePreflightStatus, CapturePreflightWait,
    CaptureReadOutcome, CaptureReceiveOutcome, CaptureRuntimeCoordinator, CaptureRuntimeWait,
    CaptureSource, CaptureSourceCapabilities, CaptureSourceFactory, CaptureSourceMetadata,
    CaptureTimestampSemantics, CaptureVideoProfile, RgbFrameEnvelope,
};
use billiards_perception::capture_pipeline::{
    AppliedCaptureCalibration, CaptureFrameRejection, CapturePipelineBindError,
    CapturePipelineConfig, CapturePipelineOutcome, CaptureScenePipeline, Rgb8FrameDetector,
};
use billiards_perception::detection_batch::{
    BatchTimestamp, ImageSpace, InferenceSettings, ModelClass, ModelIdentity, SourceFrame,
};
use billiards_perception::framing::{read_framed_message, write_framed_message};
use billiards_perception::inference::{Detection, FrameInferenceOutput, FrameInferenceTimings};
use billiards_perception::live_config::CaptureSessionCompatibility;
use billiards_perception::live_config::{CaptureControlRequirements, ExpectedCaptureSettings};
use billiards_perception::tracking::TrackingConfigV1;
use billiards_perception::ImageSpace as SceneImageSpace;
use billiards_perception::{SceneTransportConfig, TimestampStage};
use serde_json::{json, Value};

const FRAME_LIMIT_BYTES: usize = 16 * 16 * 3;
const MAXIMUM_MESSAGE_BYTES: usize = 64 * 1024;
static CAPTURE_RUNTIME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct TestDetector {
    model: ModelIdentity,
    settings: InferenceSettings,
    calls: Arc<AtomicUsize>,
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
        Ok(FrameInferenceOutput {
            source_width_px: width_px,
            source_height_px: height_px,
            detections: vec![Detection {
                class_id: 1,
                class_name: "ball".to_owned(),
                confidence: 0.95,
                bbox_xyxy_px: [3.0, 4.0, 9.0, 10.0],
                center_xy_px: [6.0, 7.0],
            }],
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

struct TestSourceFactory {
    interrupted: Arc<AtomicBool>,
    releases: Arc<AtomicUsize>,
    metadata: CaptureSourceMetadata,
}

impl CaptureSourceFactory for TestSourceFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        Arc::new(TestInterrupt(Arc::clone(&self.interrupted)))
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        Ok(Box::new(TestSource {
            interrupted: Arc::clone(&self.interrupted),
            releases: Arc::clone(&self.releases),
            emitted: false,
            metadata: self.metadata,
        }))
    }
}

struct TestSource {
    interrupted: Arc<AtomicBool>,
    releases: Arc<AtomicUsize>,
    emitted: bool,
    metadata: CaptureSourceMetadata,
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
        if self.emitted || self.interrupted.load(Ordering::Acquire) {
            return Ok(CaptureReadOutcome::End);
        }
        self.emitted = true;
        let mut pixels = frame_budget
            .allocate_rgb8_buffer(16, 16, frame_limit)
            .map_err(|error| error.to_string())?;
        pixels.as_mut_slice().fill(127);
        let frame = RgbFrameEnvelope::new(
            billiards_perception::detection_batch::SourceFrame {
                session_id: session_id.to_owned(),
                frame_id: "synthetic-frame-1".to_owned(),
                // This test covers the honest read-completion fallback. No
                // source timestamp is invented or mislabeled as exposure.
                timestamp: None,
                width_px: 16,
                height_px: 16,
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

fn source_metadata() -> CaptureSourceMetadata {
    let format = CapturePixelFormat::new("RGB8").unwrap();
    let profile = CaptureVideoProfile::new(
        16,
        16,
        CaptureFrameRate::new(30, 1).unwrap(),
        format.clone(),
    )
    .unwrap();
    let capabilities = CaptureSourceCapabilities::new(
        Some(vec![format]),
        CaptureCapability::Unknown,
        CaptureCapability::Unknown,
        CaptureCapability::Unknown,
        CaptureCapability::Unsupported,
    )
    .unwrap();
    CaptureSourceMetadata::new(
        "synthetic-source",
        Some("deterministic test source".to_owned()),
        "integration-test",
        profile.clone(),
        profile,
        capabilities,
        CaptureAppliedFrameGeometry::new(
            1,
            "synthetic-camera-profile",
            billiards_perception::ImageSpace::Raw,
            0,
        )
        .unwrap(),
        CaptureAppliedCameraControls::unknown(),
        CaptureTimestampSemantics::ReadCompletionOnly,
    )
    .unwrap()
}

fn detector() -> TestDetector {
    detector_with_calls(Arc::new(AtomicUsize::new(0)))
}

fn detector_with_calls(calls: Arc<AtomicUsize>) -> TestDetector {
    TestDetector {
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
        calls,
    }
}

fn tracking() -> TrackingConfigV1 {
    TrackingConfigV1::new(
        1,
        Some(0),
        "camera-main",
        40.0,
        Duration::from_millis(100),
        Duration::from_secs(1),
        Duration::from_secs(3),
        32,
        8,
        1,
        1,
    )
    .unwrap()
}

fn validated_calibration() -> ValidatedTableCalibrationV1 {
    let mut bundle: TableCalibrationV1 = serde_json::from_str(include_str!(
        "../contracts/calibration/v1/table-calibration-center-origin.json"
    ))
    .unwrap();
    bundle.camera_profile_id = "camera-main".to_owned();
    bundle.rectification_revision = 1;
    bundle.image_space = SceneImageSpace::Raw;
    bundle.source_width_px = 16;
    bundle.source_height_px = 16;
    bundle.pixel_to_table_mm = [[1.0, 0.0, -8.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]];
    bundle.into_validated().unwrap()
}

fn applied_calibration() -> AppliedCaptureCalibration {
    applied_calibration_with_revision(3)
}

fn applied_calibration_with_revision(revision: u64) -> AppliedCaptureCalibration {
    let mut bundle: TableCalibrationV1 = serde_json::from_str(include_str!(
        "../contracts/calibration/v1/table-calibration-center-origin.json"
    ))
    .unwrap();
    bundle.camera_profile_id = "camera-main".to_owned();
    bundle.rectification_revision = 1;
    bundle.image_space = SceneImageSpace::Raw;
    bundle.source_width_px = 16;
    bundle.source_height_px = 16;
    bundle.calibration_revision = revision;
    bundle.pixel_to_table_mm = [[1.0, 0.0, -8.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]];
    AppliedCaptureCalibration::new(bundle.into_validated().unwrap(), revision, "camera-main", 1)
        .unwrap()
}

fn expected_capture_settings(maximum_frame_age: Duration) -> ExpectedCaptureSettings {
    expected_capture_settings_at_revision(1, maximum_frame_age)
}

fn expected_capture_settings_at_revision(
    configuration_revision: u64,
    maximum_frame_age: Duration,
) -> ExpectedCaptureSettings {
    let profile = source_metadata().requested_profile().clone();
    ExpectedCaptureSettings::new(
        configuration_revision,
        "synthetic-source",
        profile.clone(),
        profile,
        "camera-main",
        SceneImageSpace::Raw,
        1,
        3,
        Some(validated_calibration()),
        CaptureTimestampSemantics::ReadCompletionOnly,
        CaptureControlRequirements::default(),
        maximum_frame_age,
        CaptureFrameLimit::new(FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(FRAME_LIMIT_BYTES * 2).unwrap(),
    )
    .unwrap()
}

fn source_metadata_for_expected(expected: &ExpectedCaptureSettings) -> CaptureSourceMetadata {
    let base = source_metadata();
    let geometry = CaptureAppliedFrameGeometry::new(
        expected.configuration_revision(),
        expected.camera_profile_id(),
        expected.image_space(),
        expected.rectification_revision(),
    )
    .unwrap();
    CaptureSourceMetadata::new(
        base.device_id(),
        base.device_name().map(str::to_owned),
        base.backend_id(),
        expected.requested_profile().clone(),
        expected.expected_negotiated_profile().clone(),
        base.capabilities().clone(),
        geometry,
        base.applied_controls().clone(),
        base.timestamp_semantics().clone(),
    )
    .unwrap()
}

fn applied_calibration_with_different_transform() -> AppliedCaptureCalibration {
    let mut bundle: TableCalibrationV1 = serde_json::from_str(include_str!(
        "../contracts/calibration/v1/table-calibration-center-origin.json"
    ))
    .unwrap();
    bundle.camera_profile_id = "camera-main".to_owned();
    bundle.rectification_revision = 1;
    bundle.image_space = SceneImageSpace::Raw;
    bundle.source_width_px = 16;
    bundle.source_height_px = 16;
    bundle.pixel_to_table_mm = [[1.0, 0.0, -7.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]];
    AppliedCaptureCalibration::new(bundle.into_validated().unwrap(), 3, "camera-main", 1).unwrap()
}

fn compatibility(expected: ExpectedCaptureSettings) -> CaptureSessionCompatibility {
    let _runtime_guard = CAPTURE_RUNTIME_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut capture = CaptureRuntimeCoordinator::new(
        CaptureFrameLimit::new(FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(FRAME_LIMIT_BYTES * 2).unwrap(),
    )
    .unwrap();
    let interrupted = Arc::new(AtomicBool::new(false));
    let releases = Arc::new(AtomicUsize::new(0));
    let metadata = source_metadata_for_expected(&expected);
    capture
        .start_preflighted(
            expected,
            TestSourceFactory {
                interrupted,
                releases,
                metadata,
            },
        )
        .unwrap();
    let status = capture
        .wait_for_preflight(Duration::from_secs(2))
        .expect("preflight state is available");
    let CapturePreflightWait::Ready(CapturePreflightStatus::Compatible(compatibility)) = status
    else {
        panic!("expected compatible source settings: {status:?}");
    };
    let compatibility = *compatibility;
    let _ = capture.stop(Duration::from_secs(2));
    compatibility
}

fn pipeline_config(
    image_space: ImageSpace,
    maximum_frame_age: Duration,
    calibration: Option<AppliedCaptureCalibration>,
) -> CapturePipelineConfig {
    CapturePipelineConfig {
        engine_instance_id: "capture-preflight-test-run".to_owned(),
        capture_configuration_revision: 1,
        image_space,
        maximum_frame_age,
        calibration,
        scene_transport: SceneTransportConfig {
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9),
            maximum_message_bytes: MAXIMUM_MESSAGE_BYTES,
            public_state_delivery_enabled: false,
            connect_timeout: Duration::from_millis(100),
            write_timeout: Duration::from_millis(100),
            reconnect_initial_delay: Duration::from_millis(10),
            reconnect_maximum_delay: Duration::from_millis(20),
        },
    }
}

fn direct_frame(
    session_id: &str,
    frame_id: &str,
    width_px: u32,
    height_px: u32,
    timestamp: Option<BatchTimestamp>,
) -> RgbFrameEnvelope {
    let frame_limit = CaptureFrameLimit::new(FRAME_LIMIT_BYTES).unwrap();
    let budget =
        CaptureFrameBudget::new(CaptureAggregateFrameLimit::new(FRAME_LIMIT_BYTES * 2).unwrap());
    let pixels = budget
        .allocate_rgb8_buffer(width_px, height_px, frame_limit)
        .unwrap();
    RgbFrameEnvelope::new(
        SourceFrame {
            session_id: session_id.to_owned(),
            frame_id: frame_id.to_owned(),
            timestamp,
            width_px,
            height_px,
        },
        pixels,
        std::time::Instant::now(),
        frame_limit,
    )
    .unwrap()
}

fn receive_one_scene(listener: TcpListener) -> Value {
    let (mut stream, _) = listener.accept().unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let hello = read_json_frame(&mut stream);
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["protocol_version"], 2);
    // The exact field set is protocol negotiation plus producer provenance;
    // it cannot silently grow a credential or challenge field.
    let hello_fields = hello
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        hello_fields,
        BTreeSet::from([
            "engine_instance_id",
            "protocol_version",
            "source_session_id",
            "type",
        ])
    );
    write_json_frame(
        &mut stream,
        &json!({
            "type": "ready",
            "protocol_version": 2,
            "last_transport_sequence": null
        }),
    );
    let bytes = read_framed_message(&mut stream, MAXIMUM_MESSAGE_BYTES).unwrap();
    let snapshot = billiards_perception::decode_scene_snapshot_v1(&bytes).unwrap();
    serde_json::to_value(snapshot).unwrap()
}

fn read_json_frame(stream: &mut TcpStream) -> Value {
    let bytes = read_framed_message(stream, MAXIMUM_MESSAGE_BYTES).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn write_json_frame(stream: &mut TcpStream, value: &Value) {
    let payload = serde_json::to_vec(value).unwrap();
    write_framed_message(stream, &payload, MAXIMUM_MESSAGE_BYTES).unwrap();
    stream.flush().unwrap();
}

#[test]
fn fake_capture_to_anonymous_loopback_scene_delivery_runs_end_to_end() {
    let _runtime_guard = CAPTURE_RUNTIME_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let receiver = thread::spawn(move || receive_one_scene(listener));

    let mut pipeline = CaptureScenePipeline::new(
        detector(),
        tracking(),
        CapturePipelineConfig {
            engine_instance_id: "capture-pipeline-integration-run".to_owned(),
            capture_configuration_revision: 1,
            image_space: ImageSpace::Raw,
            maximum_frame_age: Duration::from_secs(2),
            calibration: Some(applied_calibration()),
            scene_transport: SceneTransportConfig {
                address,
                maximum_message_bytes: MAXIMUM_MESSAGE_BYTES,
                public_state_delivery_enabled: false,
                connect_timeout: Duration::from_secs(1),
                write_timeout: Duration::from_secs(1),
                reconnect_initial_delay: Duration::from_millis(10),
                reconnect_maximum_delay: Duration::from_millis(50),
            },
        },
    )
    .unwrap();

    let mut capture = CaptureRuntimeCoordinator::new(
        CaptureFrameLimit::new(FRAME_LIMIT_BYTES).unwrap(),
        CaptureAggregateFrameLimit::new(FRAME_LIMIT_BYTES * 2).unwrap(),
    )
    .unwrap();
    let interrupted = Arc::new(AtomicBool::new(false));
    let releases = Arc::new(AtomicUsize::new(0));
    let expected = expected_capture_settings(Duration::from_secs(2));
    let metadata = source_metadata_for_expected(&expected);
    let session_id = capture
        .start_preflighted(
            expected,
            TestSourceFactory {
                interrupted,
                releases: Arc::clone(&releases),
                metadata,
            },
        )
        .unwrap();
    let compatibility = match capture
        .wait_for_preflight(Duration::from_secs(2))
        .expect("preflight state is available")
    {
        CapturePreflightWait::Ready(CapturePreflightStatus::Compatible(compatibility)) => {
            *compatibility
        }
        status => panic!("expected compatible source settings: {status:?}"),
    };
    pipeline.bind_preflight(compatibility).unwrap();
    let mut frame_receiver = capture.claim_receiver().unwrap();

    let first = pipeline.receive_one(&mut frame_receiver, Duration::from_secs(2));
    let CaptureReceiveOutcome::Frame(Ok(CapturePipelineOutcome::Published(result))) = first else {
        panic!("expected one complete scene from the synthetic capture source: {first:?}");
    };
    assert_eq!(result.scene.source.session_id, session_id);
    assert_eq!(result.scene.source.frame_id, "synthetic-frame-1");
    assert_eq!(
        result.scene.source.timestamp.stage,
        TimestampStage::CameraReadComplete
    );
    assert_eq!(result.scene.balls.len(), 1);
    assert_eq!(result.scene.revisions.calibration, Some(3));
    assert_eq!(
        result.scene.revisions.calibration_state,
        billiards_perception::CalibrationState::Valid
    );
    let measurement = result.scene.balls[0].measurement.as_ref().unwrap();
    assert_eq!(measurement.image_center_px.x, 6.0);
    assert_eq!(measurement.image_center_px.y, 7.0);
    let table_position = measurement.table_center_mm.as_ref().unwrap();
    assert_eq!(table_position.x, -2.0);
    assert_eq!(table_position.y, -1.0);
    assert!(result.calibration_ms.is_some_and(f64::is_finite));

    assert!(matches!(
        pipeline.receive_one(&mut frame_receiver, Duration::from_secs(2)),
        CaptureReceiveOutcome::Closed
    ));
    let CaptureRuntimeWait::Completed(report) = capture.wait_timeout(Duration::from_secs(2)) else {
        panic!("synthetic capture worker did not finish naturally");
    };
    assert_eq!(report.frames_published, 1);
    assert_eq!(releases.load(Ordering::SeqCst), 1);

    let transport_stats = pipeline.shutdown();
    assert_eq!(transport_stats.written_snapshots, 1);
    let received_scene = receiver.join().unwrap();
    assert_eq!(
        received_scene["engine_instance_id"],
        "capture-pipeline-integration-run"
    );
    assert_eq!(received_scene["source"]["session_id"], session_id);
    assert_eq!(received_scene["source"]["frame_id"], "synthetic-frame-1");
    assert_eq!(received_scene["outcome"], "success");
    assert_eq!(received_scene["revisions"]["calibration"], 3);
    assert_eq!(received_scene["revisions"]["calibration_state"], "valid");
    assert_eq!(
        received_scene["balls"][0]["measurement"]["table_center_mm"]["x"],
        -2.0
    );
    assert_eq!(
        received_scene["balls"][0]["measurement"]["table_center_mm"]["y"],
        -1.0
    );
}

#[test]
fn preflight_bound_pipeline_rejects_session_geometry_and_timestamp_before_inference() {
    let compatibility = compatibility(expected_capture_settings(Duration::from_secs(2)));
    let unexpected_timestamp = BatchTimestamp {
        ticks: "1".to_owned(),
        unit: billiards_perception::detection_batch::TimeUnit::Nanoseconds,
        clock: billiards_perception::detection_batch::ClockKind::Sensor,
        domain_id: "unexpected-clock".to_owned(),
        stage: billiards_perception::detection_batch::TimestampStage::SharedMemoryPublication,
    };
    let rejected_frames = [
        (
            direct_frame(
                "different-source-session",
                "wrong-session",
                16,
                16,
                None,
            ),
            CaptureFrameRejection::SourceSessionMismatch {
                expected_session_id: compatibility.session_id().to_owned(),
                actual_session_id: "different-source-session".to_owned(),
            },
        ),
        (
            direct_frame(
                compatibility.session_id(),
                "wrong-dimensions",
                8,
                32,
                None,
            ),
            CaptureFrameRejection::FrameDimensionsMismatch {
                expected_width_px: 16,
                expected_height_px: 16,
                actual_width_px: 8,
                actual_height_px: 32,
            },
        ),
        (
            direct_frame(
                compatibility.session_id(),
                "wrong-timestamp-semantics",
                16,
                16,
                Some(unexpected_timestamp),
            ),
            CaptureFrameRejection::TimestampSemanticsMismatch {
                expected: CaptureTimestampSemantics::ReadCompletionOnly,
                actual: Some(CaptureTimestampSemantics::SourceTimestamp {
                    unit: billiards_perception::detection_batch::TimeUnit::Nanoseconds,
                    clock: billiards_perception::detection_batch::ClockKind::Sensor,
                    domain_id: "unexpected-clock".to_owned(),
                    stage: billiards_perception::detection_batch::TimestampStage::SharedMemoryPublication,
                }),
            },
        ),
    ];

    for (frame, expected_rejection) in rejected_frames {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline = CaptureScenePipeline::new(
            detector_with_calls(Arc::clone(&calls)),
            tracking(),
            pipeline_config(ImageSpace::Raw, Duration::from_secs(2), None),
        )
        .unwrap();
        pipeline.bind_preflight(compatibility.clone()).unwrap();

        assert_eq!(
            pipeline.process(&frame).unwrap(),
            CapturePipelineOutcome::Rejected(expected_rejection)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(pipeline.stats().preflight_rejections, 1);
        assert_eq!(pipeline.stats().inference_started, 0);
        assert_eq!(pipeline.stats().scenes_submitted, 0);
        assert_eq!(pipeline.transport_stats().queued_snapshots, 0);
        pipeline.shutdown();
    }
}

#[test]
fn preflight_binding_requires_exact_image_space_freshness_and_calibration() {
    let validated_session = compatibility(expected_capture_settings(Duration::from_secs(2)));
    let mismatches = [
        (
            pipeline_config(ImageSpace::Rectified, Duration::from_secs(2), None),
            CapturePipelineBindError::ImageSpaceMismatch,
        ),
        (
            pipeline_config(ImageSpace::Raw, Duration::from_secs(1), None),
            CapturePipelineBindError::MaximumFrameAgeMismatch,
        ),
        (
            pipeline_config(
                ImageSpace::Raw,
                Duration::from_secs(2),
                Some(applied_calibration_with_revision(4)),
            ),
            CapturePipelineBindError::CalibrationMismatch(
                "runtime calibration revision differs from the expected revision",
            ),
        ),
        (
            pipeline_config(
                ImageSpace::Raw,
                Duration::from_secs(2),
                Some(applied_calibration_with_different_transform()),
            ),
            CapturePipelineBindError::CalibrationMismatch(
                "runtime calibration bundle differs from the expected bundle",
            ),
        ),
    ];

    for (config, expected_error) in mismatches {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut pipeline =
            CaptureScenePipeline::new(detector_with_calls(Arc::clone(&calls)), tracking(), config)
                .unwrap();
        assert_eq!(
            pipeline
                .bind_preflight(validated_session.clone())
                .unwrap_err(),
            expected_error
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(pipeline.stats().frames_received, 0);
        assert_eq!(pipeline.stats().inference_started, 0);
        assert_eq!(pipeline.stats().scenes_submitted, 0);
        assert_eq!(pipeline.transport_stats().queued_snapshots, 0);
        pipeline.shutdown();
    }

    let revision_two = compatibility(expected_capture_settings_at_revision(
        2,
        Duration::from_secs(2),
    ));
    let mut pipeline = CaptureScenePipeline::new(
        detector(),
        tracking(),
        pipeline_config(ImageSpace::Raw, Duration::from_secs(2), None),
    )
    .unwrap();
    assert_eq!(
        pipeline.bind_preflight(revision_two).unwrap_err(),
        CapturePipelineBindError::ConfigurationRevisionMismatch
    );
    pipeline.shutdown();

    // The scene tracker carries its own tracking configuration revision; the
    // capture binding uses the explicitly configured runtime revision.
    let revision_two = compatibility(expected_capture_settings_at_revision(
        2,
        Duration::from_secs(2),
    ));
    let mut config = pipeline_config(ImageSpace::Raw, Duration::from_secs(2), None);
    config.capture_configuration_revision = 2;
    let mut matching_revision_pipeline =
        CaptureScenePipeline::new(detector(), tracking(), config).unwrap();
    matching_revision_pipeline
        .bind_preflight(revision_two)
        .unwrap();
    matching_revision_pipeline.shutdown();
}

#[test]
fn missing_runtime_calibration_keeps_table_coordinates_unavailable() {
    let validated_session = compatibility(expected_capture_settings(Duration::from_secs(2)));
    let mut pipeline = CaptureScenePipeline::new(
        detector(),
        tracking(),
        pipeline_config(ImageSpace::Raw, Duration::from_secs(2), None),
    )
    .unwrap();
    pipeline.bind_preflight(validated_session.clone()).unwrap();
    let frame = direct_frame(
        validated_session.session_id(),
        "without-runtime-calibration",
        16,
        16,
        None,
    );

    let CapturePipelineOutcome::Published(result) = pipeline.process(&frame).unwrap() else {
        panic!("matching frame should publish without a runtime calibration");
    };
    assert_eq!(result.scene.revisions.calibration, None);
    assert_eq!(
        result.scene.revisions.calibration_state,
        billiards_perception::CalibrationState::Unavailable
    );
    assert!(result.scene.balls[0]
        .measurement
        .as_ref()
        .unwrap()
        .table_center_mm
        .is_none());
    pipeline.shutdown();
}

#[test]
fn validated_binding_is_one_time_and_cannot_be_added_after_processing() {
    let validated_session = compatibility(expected_capture_settings(Duration::from_secs(2)));
    let different_revision_and_session = compatibility(expected_capture_settings_at_revision(
        2,
        Duration::from_secs(2),
    ));
    assert_ne!(
        validated_session.session_id(),
        different_revision_and_session.session_id()
    );

    let mut bound = CaptureScenePipeline::new(
        detector(),
        tracking(),
        pipeline_config(ImageSpace::Raw, Duration::from_secs(2), None),
    )
    .unwrap();
    bound.bind_preflight(validated_session.clone()).unwrap();
    assert_eq!(
        bound
            .bind_preflight(different_revision_and_session)
            .unwrap_err(),
        CapturePipelineBindError::AlreadyBound
    );
    bound.shutdown();

    let mut unbound = CaptureScenePipeline::new(
        detector(),
        tracking(),
        pipeline_config(ImageSpace::Raw, Duration::from_secs(2), None),
    )
    .unwrap();
    let frame = direct_frame("legacy-replay-session", "replay-before-bind", 16, 16, None);
    assert!(matches!(
        unbound.process(&frame).unwrap(),
        CapturePipelineOutcome::Published(_)
    ));
    assert_eq!(
        unbound.bind_preflight(validated_session).unwrap_err(),
        CapturePipelineBindError::FramesAlreadyProcessed
    );
    unbound.shutdown();
}
