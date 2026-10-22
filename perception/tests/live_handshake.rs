use billiards_perception::capture_settings_wire::CaptureSettingsSnapshotV3;
use billiards_perception::live_handshake::validate_live_v3_hello;
use serde_json::{json, Value};

const CAPTURE_SETTINGS_V3: &str =
    include_str!("../contracts/capture-settings/v3/capture-settings-default.json");
const ACKNOWLEDGEMENT_V2: &str =
    include_str!("../contracts/capture-session-ack/v2/capture-session-ack-default.json");

fn settings() -> billiards_perception::capture_settings_wire::ResolvedCaptureSettings {
    CaptureSettingsSnapshotV3::parse_json(CAPTURE_SETTINGS_V3).unwrap()
}

fn hello() -> Value {
    let settings = settings();
    let mut acknowledgement: Value = serde_json::from_str(ACKNOWLEDGEMENT_V2).unwrap();
    acknowledgement["configuration_revision"] = json!(settings.configuration_revision);
    acknowledgement["request_sha256"] = json!(settings.request_sha256);
    acknowledgement["source_report"]["session_id"] = json!("capture-session-live-a");
    acknowledgement["source_report"]["applied_frame_geometry"]["camera_profile_id"] =
        json!(settings.expected.camera_profile_id());

    json!({
        "type": "hello",
        "protocol_version": 3,
        "engine_instance_id": "engine-live-a",
        "source_session_id": "capture-session-live-a",
        "capture_session_ack": acknowledgement,
    })
}

#[test]
fn compatible_live_v3_hello_is_bound_to_identity_request_and_source_report() {
    let settings = settings();
    let encoded = serde_json::to_vec(&hello()).unwrap();

    let validated = validate_live_v3_hello(&encoded, &settings).unwrap();

    assert_eq!(validated.identity.engine_instance_id, "engine-live-a");
    assert_eq!(
        validated.identity.source_session_id,
        "capture-session-live-a"
    );
    assert_eq!(
        validated.acknowledgement.configuration_revision,
        settings.configuration_revision
    );
    assert_eq!(
        validated.acknowledgement.request_sha256,
        settings.request_sha256
    );
    assert_eq!(
        validated.compatibility.session_id(),
        "capture-session-live-a"
    );
    assert_eq!(
        validated.acknowledgement.source_report["applied_controls"]["exposure_value"]["status"],
        "unknown",
        "validation must preserve source-reported unavailable values"
    );
}

#[test]
fn stale_configuration_revision_is_rejected() {
    let mut value = hello();
    value["capture_session_ack"]["configuration_revision"] = json!(2);

    assert!(validate_live_v3_hello(&serde_json::to_vec(&value).unwrap(), &settings()).is_err());
}

#[test]
fn request_digest_mismatch_is_rejected() {
    let mut value = hello();
    value["capture_session_ack"]["request_sha256"] = json!("0".repeat(64));

    assert!(validate_live_v3_hello(&serde_json::to_vec(&value).unwrap(), &settings()).is_err());
}

#[test]
fn acknowledgement_must_name_the_hello_source_session() {
    let mut value = hello();
    value["capture_session_ack"]["source_report"]["session_id"] = json!("other-session");

    assert!(validate_live_v3_hello(&serde_json::to_vec(&value).unwrap(), &settings()).is_err());
}

#[test]
fn unknown_hello_and_report_fields_are_rejected() {
    let mut hello_extra = hello();
    hello_extra["unexpected"] = json!(true);
    assert!(
        validate_live_v3_hello(&serde_json::to_vec(&hello_extra).unwrap(), &settings()).is_err()
    );

    let mut report_extra = hello();
    report_extra["capture_session_ack"]["source_report"]["camera_applied"] = json!(true);
    assert!(
        validate_live_v3_hello(&serde_json::to_vec(&report_extra).unwrap(), &settings()).is_err()
    );
}

#[test]
fn source_report_semantics_must_match_resolved_capture_settings() {
    let mut value = hello();
    value["capture_session_ack"]["source_report"]["device_id"] =
        json!("device-index:another-camera");

    assert!(validate_live_v3_hello(&serde_json::to_vec(&value).unwrap(), &settings()).is_err());
}

#[test]
fn non_compatible_acknowledgement_cannot_make_the_producer_ready() {
    let mut value = hello();
    value["capture_session_ack"]["outcome"] = json!("rejected");
    value["capture_session_ack"]["failure"] = json!({ "category": "preflight_rejected" });

    assert!(validate_live_v3_hello(&serde_json::to_vec(&value).unwrap(), &settings()).is_err());
}
