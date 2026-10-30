//! Process-level checks for the production live-application boundary.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use billiards_perception::capture::{
    CaptureInterrupt, CaptureOpenError, CaptureSource, CaptureSourceFactory,
};
use billiards_perception::capture_settings_wire::CaptureSettingsSnapshotV1;
use billiards_perception::inference::ProviderSelection;
use billiards_perception::live_application::{
    parse_phase_schedule_json, run_live_application, LiveApplicationSettings,
};
use billiards_perception::SceneTransportConfig;
use serde_json::{json, Value};

const MAXIMUM_SCENE_BYTES: usize = 64 * 1024;

#[test]
fn live_application_rejects_cpu_provider_before_capture_is_opened() {
    let request = test_request_json();
    let resolved = CaptureSettingsSnapshotV1::parse_json(&request).unwrap();
    let settings = live_settings(
        std::path::PathBuf::new(),
        std::path::PathBuf::new(),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
        &resolved.expected,
    );
    let shutdown_requested = AtomicBool::new(false);

    let error = run_live_application(&request, settings, NeverOpenedFactory, &shutdown_requested)
        .unwrap_err();

    assert!(error.to_string().contains("requires the OpenVINO GPU"));
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

struct NeverOpenedFactory;

impl CaptureSourceFactory for NeverOpenedFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        panic!("the provider check must run before capture startup")
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        panic!("the provider check must run before opening the capture source")
    }
}
