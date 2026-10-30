//! Production live composition rejects non-GPU providers before touching capture.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
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

struct NeverOpenedFactory;

impl CaptureSourceFactory for NeverOpenedFactory {
    fn interrupt_handle(&self) -> Arc<dyn CaptureInterrupt> {
        panic!("rejected live provider must not touch capture")
    }

    fn open(self: Box<Self>) -> Result<Box<dyn CaptureSource>, CaptureOpenError> {
        panic!("rejected live provider must not open capture")
    }
}

#[test]
fn production_live_application_rejects_cpu_before_capture_or_model_startup() {
    let request = include_str!("../contracts/capture-settings/v1/capture-settings-default.json");
    let expected = CaptureSettingsSnapshotV1::parse_json(request).unwrap();
    let settings = LiveApplicationSettings {
        onnx_path: None,
        source_checkpoint_path: PathBuf::new(),
        provider: ProviderSelection::Cpu,
        image_size: 640,
        confidence: 0.4,
        iou: 0.45,
        max_detections: 300,
        intra_op_threads: None,
        warmup_runs: 0,
        source_image_space: expected.expected.image_space(),
        engine_instance_id: "test-engine".to_owned(),
        source_session_id: "test-session".to_owned(),
        tracking_ball_class_id: 1,
        tracking_cue_class_id: Some(0),
        tracking_camera_profile_id: expected.expected.camera_profile_id().to_owned(),
        tracking_association_gate_px: 40.0,
        tracking_association_prediction_horizon: Duration::from_millis(100),
        tracking_lost_after: Duration::from_secs(1),
        tracking_maximum_track_age: Duration::from_secs(3),
        tracking_maximum_active_tracks: 300,
        tracking_recent_observation_capacity: 8,
        tracking_configuration_revision: expected.expected.configuration_revision(),
        tracking_rectification_revision: expected.expected.rectification_revision(),
        phase_schedule: parse_phase_schedule_json(
            r#"{"recovering":{"motion_hz":15,"ball_hz":5,"cue_hz":5},"pre_shot":{"motion_hz":15,"ball_hz":1,"cue_hz":5},"aiming":{"motion_hz":15,"ball_hz":1,"cue_hz":5},"shot":{"motion_hz":15,"ball_hz":0,"cue_hz":0},"quiet_dwell_ms":800,"layout_consensus_count":3,"layout_position_tolerance_px":4.0,"motion_evidence_expiry_ms":200,"cue_evidence_expiry_ms":250,"layout_evidence_expiry_ms":1500}"#,
        )
        .unwrap(),
        table_visibility:
            billiards_perception::table_visibility::TableVisibilityConfig::default(),
        ball_localization:
            billiards_perception::ball_localization_profile::BallLocalizationSettings::default(),
        scene_transport: SceneTransportConfig {
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            maximum_message_bytes: 64 * 1024,
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
    };

    let error = run_live_application(
        request,
        settings,
        NeverOpenedFactory,
        &AtomicBool::new(false),
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("requires the OpenVINO GPU inference provider"));
}
