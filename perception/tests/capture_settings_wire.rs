use std::time::Duration;

use billiards_perception::capture::CaptureExposureMode;
use billiards_perception::capture_settings_wire::{
    parse_capture_settings_json, CaptureSettingsSnapshotV1, CaptureSettingsSnapshotV2,
    CaptureSettingsSnapshotV3, ExposureModeV1, ResolvedCameraControlsPolicy,
};
use billiards_perception::live_config::RequestedCaptureControlValues;
use serde_json::Value;
use sha2::{Digest, Sha256};

const CAPTURE_SETTINGS_V1: &str =
    include_str!("../contracts/capture-settings/v1/capture-settings-default.json");
const CAPTURE_SETTINGS_WITH_CALIBRATION_V1: &str =
    include_str!("../contracts/capture-settings/v1/capture-settings-with-calibration.json");
const CAPTURE_SETTINGS_V2: &str =
    include_str!("../contracts/capture-settings/v2/capture-settings-default.json");
const CAPTURE_SETTINGS_V2_TRANSFORMED: &str =
    include_str!("../contracts/capture-settings/v2/capture-settings-transformed-output.json");
const CAPTURE_SETTINGS_V3: &str =
    include_str!("../contracts/capture-settings/v3/capture-settings-default.json");
const CAPTURE_SETTINGS_V3_REQUIRE: &str =
    include_str!("../contracts/capture-settings/v3/capture-settings-require-controls.json");

#[test]
fn capture_settings_fixture_resolves_to_validated_rust_expectations() {
    let resolved = CaptureSettingsSnapshotV1::parse_json(CAPTURE_SETTINGS_V1).unwrap();

    assert_eq!(resolved.configuration_revision, 1);
    assert_eq!(resolved.requested_backend, "auto");
    assert_eq!(resolved.expected.requested_backend(), Some("auto"));
    assert_eq!(resolved.expected.configuration_revision(), 1);
    assert_eq!(resolved.expected.device_id(), "device-index:0");
    assert_eq!(resolved.expected.requested_profile().width_px(), 1920);
    assert_eq!(resolved.expected.requested_profile().height_px(), 1080);
    assert_eq!(
        resolved
            .expected
            .requested_profile()
            .frame_rate()
            .frames_numerator(),
        30
    );
    assert_eq!(
        resolved.expected.maximum_frame_age(),
        Duration::from_millis(100)
    );
    let ResolvedCameraControlsPolicy::Require { requested, .. } = &resolved.camera_controls else {
        panic!("v1 must retain its strict controls policy");
    };
    assert_eq!(requested.exposure_mode, ExposureModeV1::Auto);
    assert_eq!(requested.gain, 1.0);
    assert_eq!(
        resolved.expected.requested_control_values(),
        Some(RequestedCaptureControlValues {
            exposure_mode: CaptureExposureMode::Automatic,
            exposure_value: None,
            gain: 1.0,
        })
    );
    assert_eq!(requested.brightness, 0.5);
    assert_eq!(requested.contrast, 0.5);
    assert_eq!(requested.saturation, 0.5);
    assert!(requested.auto_focus);
    assert_eq!(requested.buffer_size, 2);
    assert!(resolved.reconnect_policy.enabled);
    assert_eq!(resolved.reconnect_policy.delay_seconds, 1.0);
    assert_eq!(resolved.reconnect_policy.maximum_attempts, 5);
    assert_eq!(resolved.reconnect_policy.stop_timeout_seconds, 5.0);
}

#[test]
fn request_digest_matches_deterministic_json_and_preserves_input_bytes() {
    let request: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    // serde_json's Value object uses sorted keys, producing the canonical
    // compact request encoding used by this ASCII-only fixture.
    let canonical_json = serde_json::to_string(&request).unwrap();
    let resolved = CaptureSettingsSnapshotV1::parse_json(&canonical_json).unwrap();
    let acknowledgement: Value = serde_json::from_str(include_str!(
        "../contracts/capture-session-ack/v1/capture-session-ack-default.json"
    ))
    .unwrap();
    assert_eq!(
        resolved.request_sha256,
        acknowledgement["request_sha256"].as_str().unwrap()
    );

    let parsed_with_original_whitespace =
        CaptureSettingsSnapshotV1::parse_json(CAPTURE_SETTINGS_V1).unwrap();
    assert_eq!(
        parsed_with_original_whitespace.request_sha256,
        format!("{:x}", Sha256::digest(CAPTURE_SETTINGS_V1.as_bytes()))
    );
    assert_ne!(
        parsed_with_original_whitespace.request_sha256,
        resolved.request_sha256
    );
}

#[test]
fn wire_request_preserves_manual_exposure_values_for_source_readback_checks() {
    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["requested_controls"]["exposure_mode"] = serde_json::json!("manual");
    value["requested"]["requested_controls"]["exposure_value"] = serde_json::json!(-4.0);
    value["requested"]["control_requirements"]["manual_exposure"] = serde_json::json!(true);
    value["requested"]["control_requirements"]["automatic_exposure"] = serde_json::json!(false);

    let resolved = CaptureSettingsSnapshotV1::parse_json(&value.to_string()).unwrap();
    let ResolvedCameraControlsPolicy::Require { requested, .. } = &resolved.camera_controls else {
        panic!("v1 must retain its strict controls policy");
    };
    assert_eq!(
        resolved.expected.requested_control_values(),
        Some(RequestedCaptureControlValues {
            exposure_mode: CaptureExposureMode::Manual,
            exposure_value: Some(-4.0),
            gain: 1.0,
        })
    );
    assert_eq!(requested.exposure_value, Some(-4.0));
}

#[test]
fn calibration_bearing_capture_fixture_resolves_to_validated_rust_calibration() {
    let resolved =
        CaptureSettingsSnapshotV1::parse_json(CAPTURE_SETTINGS_WITH_CALIBRATION_V1).unwrap();
    let calibration = resolved
        .expected
        .calibration()
        .expect("shared fixture must carry a validated calibration");

    assert_eq!(
        calibration.calibration_revision(),
        resolved.expected.calibration_revision()
    );
    assert_eq!(
        calibration.camera_profile_id(),
        resolved.expected.camera_profile_id()
    );
}

#[test]
fn wire_contract_rejects_unknown_fields_instead_of_ignoring_new_semantics() {
    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["applied"] = serde_json::json!({"gain": 1.0});
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["requested_profile"]["driver_magic"] = serde_json::json!(true);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());
}

#[test]
fn wire_contract_rejects_unsupported_versions_missing_fields_and_bad_profiles() {
    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["schema_version"] = serde_json::json!(2);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]
        .as_object_mut()
        .unwrap()
        .remove("calibration");
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["expected_negotiated_profile"]["width_px"] = serde_json::json!(0);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());
}

#[test]
fn wire_contract_rejects_inconsistent_controls_limits_and_freshness() {
    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["requested_controls"]["exposure_mode"] = serde_json::json!("manual");
    value["requested"]["control_requirements"]["manual_exposure"] = serde_json::json!(true);
    value["requested"]["control_requirements"]["automatic_exposure"] = serde_json::json!(false);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["maximum_rgb_frame_bytes"] = serde_json::json!(1);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["maximum_aggregate_rgb_frame_bytes"] = serde_json::json!(1);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["maximum_frame_age_ms"] = serde_json::json!(0);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());
}

#[test]
fn wire_contract_rejects_values_outside_the_schema_resource_bounds() {
    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["maximum_frame_age_ms"] = serde_json::json!(9_007_199_254_740_992_u64);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["maximum_rgb_frame_bytes"] = serde_json::json!(1_073_741_825_u64);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["maximum_aggregate_rgb_frame_bytes"] = serde_json::json!(1_073_741_825_u64);
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());
}

#[test]
fn wire_contract_rejects_zero_configuration_revision() {
    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["configuration_revision"] = serde_json::json!(0);

    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());
}

#[test]
fn wire_contract_rejects_unsupported_backends_and_unsupported_identifiers() {
    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["backend"] = serde_json::json!("usb-camera");
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["device_id"] = serde_json::json!("device/index:0");
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());

    let mut value: Value = serde_json::from_str(CAPTURE_SETTINGS_V1).unwrap();
    value["requested"]["camera_profile_id"] = serde_json::json!("camera-profile-café");
    assert!(CaptureSettingsSnapshotV1::parse_json(&value.to_string()).is_err());
}

#[test]
fn v2_contract_separates_negotiated_camera_mode_from_emitted_rgb8_geometry() {
    let resolved = CaptureSettingsSnapshotV2::parse_json(CAPTURE_SETTINGS_V2).unwrap();
    assert_eq!(resolved.schema_version, 2);
    assert_eq!(resolved.expected.requested_profile().width_px(), 1920);
    assert_eq!(
        resolved
            .expected
            .requested_profile()
            .pixel_format()
            .identifier(),
        "RGB8"
    );
    assert_eq!(
        resolved
            .expected
            .output_frame_geometry()
            .unwrap()
            .width_px(),
        1920
    );
    assert_eq!(
        resolved
            .expected
            .output_frame_geometry()
            .unwrap()
            .height_px(),
        1080
    );
    assert_eq!(
        parse_capture_settings_json(CAPTURE_SETTINGS_V2)
            .unwrap()
            .schema_version,
        2
    );
    assert_eq!(
        parse_capture_settings_json(CAPTURE_SETTINGS_V1)
            .unwrap()
            .schema_version,
        1
    );

    let transformed =
        CaptureSettingsSnapshotV2::parse_json(CAPTURE_SETTINGS_V2_TRANSFORMED).unwrap();
    assert_eq!(
        transformed
            .expected
            .requested_profile()
            .pixel_format()
            .identifier(),
        "MJPG"
    );
    assert_eq!(
        transformed
            .expected
            .output_frame_geometry()
            .map(|geometry| (geometry.width_px(), geometry.height_px())),
        Some((1280, 720))
    );
}

#[test]
fn v2_frame_budget_is_validated_against_emitted_rgb8_dimensions() {
    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V2_TRANSFORMED).unwrap();
    request["requested"]["maximum_rgb_frame_bytes"] = serde_json::json!(2_764_800);
    request["requested"]["maximum_aggregate_rgb_frame_bytes"] = serde_json::json!(5_529_600);
    let resolved = CaptureSettingsSnapshotV2::parse_json(&request.to_string()).unwrap();
    assert_eq!(
        resolved.expected.frame_limit().maximum_rgb8_bytes(),
        2_764_800
    );

    // The same budget is insufficient for the 1920x1080 raw RGB profile.
    request["requested"]["output_frame_geometry"]["width_px"] = serde_json::json!(1920);
    request["requested"]["output_frame_geometry"]["height_px"] = serde_json::json!(1080);
    assert!(CaptureSettingsSnapshotV2::parse_json(&request.to_string()).is_err());
}

#[test]
fn v2_contract_requires_valid_output_geometry_and_rejects_unknown_fields() {
    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V2).unwrap();
    request["requested"]
        .as_object_mut()
        .unwrap()
        .remove("output_frame_geometry");
    assert!(CaptureSettingsSnapshotV2::parse_json(&request.to_string()).is_err());

    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V2).unwrap();
    request["requested"]["output_frame_geometry"]["width_px"] = serde_json::json!(0);
    assert!(CaptureSettingsSnapshotV2::parse_json(&request.to_string()).is_err());

    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V2).unwrap();
    request["requested"]["output_frame_geometry"]["future_field"] = serde_json::json!(true);
    assert!(CaptureSettingsSnapshotV2::parse_json(&request.to_string()).is_err());
}

#[test]
fn v2_calibration_dimensions_belong_to_emitted_image_space() {
    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_WITH_CALIBRATION_V1).unwrap();
    request["schema_version"] = serde_json::json!(2);
    request["requested"]["output_frame_geometry"] =
        serde_json::json!({"width_px": 500, "height_px": 250});
    request["requested"]["calibration"]["source_width_px"] = serde_json::json!(500);
    request["requested"]["calibration"]["source_height_px"] = serde_json::json!(250);

    let resolved = CaptureSettingsSnapshotV2::parse_json(&request.to_string()).unwrap();
    assert_eq!(
        resolved
            .expected
            .calibration()
            .unwrap()
            .source_dimensions_px(),
        (500, 250)
    );

    request["requested"]["output_frame_geometry"]["width_px"] = serde_json::json!(501);
    assert!(CaptureSettingsSnapshotV2::parse_json(&request.to_string()).is_err());
}

#[test]
fn v3_source_defaults_retains_output_geometry_without_inventing_control_values() {
    let resolved = CaptureSettingsSnapshotV3::parse_json(CAPTURE_SETTINGS_V3).unwrap();
    assert_eq!(resolved.schema_version, 3);
    assert_eq!(
        resolved
            .expected
            .output_frame_geometry()
            .map(|geometry| (geometry.width_px(), geometry.height_px())),
        Some((1920, 1080))
    );
    assert_eq!(
        resolved.expected.requested_control_values(),
        None,
        "source defaults must not fabricate requested control values"
    );
    assert!(matches!(
        resolved.camera_controls,
        ResolvedCameraControlsPolicy::SourceDefaults
    ));
    assert_eq!(
        parse_capture_settings_json(CAPTURE_SETTINGS_V3)
            .unwrap()
            .schema_version,
        3
    );
}

#[test]
fn store_owned_tombstone_revision_is_visible_without_changing_exact_request_digest() {
    let mut resolved = parse_capture_settings_json(CAPTURE_SETTINGS_V3).unwrap();
    let request_digest = resolved.request_sha256.clone();

    resolved
        .expected
        .apply_calibration_store_state(12, None)
        .unwrap();

    assert_eq!(resolved.expected.calibration_revision(), 12);
    assert!(resolved.expected.calibration().is_none());
    assert_eq!(resolved.request_sha256, request_digest);
}

#[test]
fn v3_require_policy_preserves_the_strict_control_request() {
    let resolved = CaptureSettingsSnapshotV3::parse_json(CAPTURE_SETTINGS_V3_REQUIRE).unwrap();
    assert_eq!(resolved.schema_version, 3);
    assert_eq!(
        resolved.expected.requested_control_values(),
        Some(RequestedCaptureControlValues {
            exposure_mode: CaptureExposureMode::Automatic,
            exposure_value: None,
            gain: 1.0,
        })
    );
    assert!(matches!(
        resolved.camera_controls,
        ResolvedCameraControlsPolicy::Require { .. }
    ));
}

#[test]
fn v3_control_policy_rejects_unknown_modes_extra_fields_and_incomplete_requirements() {
    for policy in [
        serde_json::json!({"mode": "relaxed"}),
        serde_json::json!({"mode": "source_defaults", "requested": {}}),
        serde_json::json!({"mode": "require", "requirements": {}, "requested": {}}),
    ] {
        let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V3).unwrap();
        request["requested"]["camera_controls"] = policy;
        assert!(
            CaptureSettingsSnapshotV3::parse_json(&request.to_string()).is_err(),
            "malformed v3 policy unexpectedly parsed: {request}"
        );
    }

    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V3_REQUIRE).unwrap();
    request["requested"]["camera_controls"]["requested"]["future_control"] =
        serde_json::json!(true);
    assert!(CaptureSettingsSnapshotV3::parse_json(&request.to_string()).is_err());

    let mut request: Value = serde_json::from_str(CAPTURE_SETTINGS_V3).unwrap();
    request["requested"]["camera_controls"]["future_policy_field"] = serde_json::json!(true);
    assert!(CaptureSettingsSnapshotV3::parse_json(&request.to_string()).is_err());
}
