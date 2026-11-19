use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use billiards_perception::configuration_authority::{
    AuthorityLimits, ConfigurationAuthority, DEFAULT_MAX_CONFIG_BYTES, DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES,
};
use fs2::FileExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn authority(directory: &TempDir) -> (ConfigurationAuthority, std::path::PathBuf) {
    let config_path = directory.path().join("isolated-config.json");
    let authority = ConfigurationAuthority::new(&config_path, AuthorityLimits::default())
        .expect("valid authority config");
    (authority, config_path)
}

fn call(authority: &ConfigurationAuthority, request: Value) -> Value {
    let bytes = serde_json::to_vec(&request).expect("request serialization");
    let (response, _) = authority.handle_json(&bytes);
    serde_json::from_slice(&response).expect("response JSON")
}

fn request(operation: &str) -> Value {
    json!({"protocol_version": 1, "operation": operation})
}

#[test]
fn snapshot_schema_and_defaults_are_embedded_and_use_temporary_store_path() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, config_path) = authority(&directory);
    let snapshot = call(&authority, request("snapshot"));
    assert_eq!(snapshot["revision"], 1);
    assert!(snapshot["configuration"].get("vision").is_some());
    assert!(
        !config_path.exists(),
        "read-only snapshot should not create the store"
    );

    let schema = call(&authority, request("schema"));
    assert_eq!(schema["ok"], true);
    assert!(
        schema["schema"]["x-cross-field-rules"]
            .as_array()
            .unwrap()
            .len()
            >= 6
    );
    let defaults = call(&authority, request("defaults"));
    assert_eq!(
        defaults["configuration"]["metadata"]["configuration_revision"],
        1
    );
    let provider_values = schema["schema"]["properties"]["perception"]["properties"]["rust_live"]
        ["properties"]["provider"]["enum"]
        .as_array()
        .unwrap();
    assert_eq!(provider_values, &[json!("openvino")]);
    assert!(!provider_values.contains(&json!("coral")));
    assert_eq!(
        defaults["configuration"]["perception"]["rust_live"]["provider"],
        "openvino"
    );
    assert!(defaults["configuration"]["perception"]["rust_live"]
        .get("coral_runtime")
        .is_none());
    assert_eq!(
        defaults["configuration"]["perception"]["freshness"]["expire_after_ms"],
        2_000
    );
    assert_eq!(
        schema["schema"]["properties"]["perception"]["properties"]["freshness"]["properties"]
            ["expire_after_ms"]["minimum"],
        20
    );
}

#[test]
fn capture_settings_operation_accepts_live_defaults() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, _) = authority(&directory);

    let response = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "capture_settings",
            "data": {"schema_version": 3}
        }),
    );

    assert_eq!(response["ok"], true);
    assert_eq!(response["operation"], "capture_settings");
}

#[test]
fn capture_settings_operation_resolves_v3_defaults_and_preserves_profile_hash() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, config_path) = authority(&directory);
    let update = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"video_file_path": ""}}}
        }),
    );
    assert_eq!(update["revision"], 2);
    let before = fs::read(&config_path).unwrap();

    let response = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "capture_settings",
            "data": {"schema_version": 3}
        }),
    );

    assert_eq!(response["ok"], true);
    assert_eq!(response["operation"], "capture_settings");
    assert_eq!(response["revision"], 2);
    let mut capture_settings_v3_golden: Value = serde_json::from_str(include_str!(
        "../contracts/capture-settings/v3/capture-settings-default.json"
    ))
    .unwrap();
    capture_settings_v3_golden["configuration_revision"] = json!(2);
    assert_eq!(response["capture_settings"], capture_settings_v3_golden);
    assert_eq!(response["capture_settings"]["schema_version"], 3);
    assert_eq!(response["capture_settings"]["configuration_revision"], 2);
    let requested = &response["capture_settings"]["requested"];
    assert_eq!(requested["device_id"], "device-index:0");
    assert_eq!(requested["backend"], "auto");
    assert_eq!(
        requested["camera_profile_id"],
        "camera-profile-v3:a71c7321562c3b6e5584473d5888e3823fea6264107814fd071ea53daa1dcf5f"
    );
    assert_eq!(requested["requested_profile"]["width_px"], 1920);
    assert_eq!(requested["requested_profile"]["height_px"], 1080);
    assert_eq!(
        requested["expected_negotiated_profile"],
        requested["requested_profile"]
    );
    assert_eq!(requested["output_frame_geometry"]["width_px"], 1920);
    assert_eq!(requested["output_frame_geometry"]["height_px"], 1080);
    assert_eq!(requested["calibration"], Value::Null);
    assert_eq!(
        requested["camera_controls"],
        json!({"mode": "source_defaults"})
    );
    assert_eq!(
        requested["timestamp_semantics"],
        json!({"kind": "read_completion_only"})
    );

    let encoded = serde_json::to_string(&response["capture_settings"]).unwrap();
    assert!(!encoded.contains(": "));
    let parsed = billiards_perception::capture_settings_wire::parse_capture_settings_json(&encoded)
        .expect("generated request passes strict Rust parser");
    assert_eq!(parsed.schema_version, 3);
    assert_eq!(parsed.configuration_revision, 2);
    assert_eq!(
        parsed.request_sha256,
        format!("{:x}", Sha256::digest(encoded.as_bytes()))
    );
    assert_eq!(
        fs::read(config_path).unwrap(),
        before,
        "operation is read-only"
    );
}

#[test]
fn capture_settings_operation_preserves_selector_controls_geometry_and_limits() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, _) = authority(&directory);
    let selector = format!("gstreamer-device-v1:{}", "a".repeat(64));
    let update = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {
                "vision": {"camera": {
                    "video_file_path": "",
                    "device_id": 4,
                    "backend": "gstreamer",
                    "resolution": [640, 480],
                    "control_policy": "require",
                    "exposure_mode": "manual",
                    "exposure_value": 0.01,
                    "gain": 2.5,
                    "brightness": 0.25,
                    "contrast": 0.75,
                    "saturation": 0.5
                }},
                "perception": {"capture": {
                    "device_selector": selector,
                    "output_resolution": [320, 240],
                    "maximum_rgb_frame_bytes": 230400,
                    "maximum_aggregate_rgb_frame_bytes": 230400
                }}
            }
        }),
    );
    assert_eq!(update["revision"], 2);

    let response = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "capture_settings",
            "data": {"schema_version": 3}
        }),
    );
    assert_eq!(response["ok"], true);
    let requested = &response["capture_settings"]["requested"];
    assert_eq!(requested["device_id"], selector);
    assert_eq!(requested["backend"], "gstreamer");
    assert_eq!(
        requested["camera_profile_id"],
        "camera-profile-v3:a7db912119ff227f5b0c7809a1c35ff0a1462ca41ca5e21330ef5cc27ba6f4c0"
    );
    assert_eq!(requested["requested_profile"]["width_px"], 640);
    assert_eq!(requested["requested_profile"]["height_px"], 480);
    assert_eq!(
        requested["output_frame_geometry"],
        json!({"width_px": 320, "height_px": 240})
    );
    assert_eq!(requested["maximum_rgb_frame_bytes"], 230400);
    assert_eq!(requested["maximum_aggregate_rgb_frame_bytes"], 230400);
    assert_eq!(requested["camera_controls"]["mode"], "require");
    assert_eq!(
        requested["camera_controls"]["requirements"],
        json!({
            "manual_exposure": true,
            "automatic_exposure": false,
            "gain_control": true
        })
    );
    assert_eq!(
        requested["camera_controls"]["requested"]["exposure_mode"],
        "manual"
    );
    assert_eq!(
        requested["camera_controls"]["requested"]["exposure_value"],
        0.01
    );
    assert_eq!(requested["camera_controls"]["requested"]["gain"], 2.5);
    assert_eq!(
        requested["camera_controls"]["requested"]["brightness"],
        0.25
    );
}

#[test]
fn capture_settings_operation_rejects_unsupported_schema_and_oversized_rgb_frames() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, _) = authority(&directory);
    let unsupported = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "capture_settings",
            "data": {"schema_version": 2}
        }),
    );
    assert_eq!(
        unsupported["error"]["code"],
        "unsupported_capture_settings_version"
    );

    let update = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {
                "vision": {"camera": {"video_file_path": ""}},
                "perception": {"capture": {"output_resolution": [4096, 4096]}}
            }
        }),
    );
    assert_eq!(update["revision"], 2);
    let too_large = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "capture_settings",
            "data": {"schema_version": 3}
        }),
    );
    assert_eq!(too_large["error"]["code"], "invalid_capture_settings");
    assert!(too_large["error"]["message"]
        .as_str()
        .unwrap()
        .contains("exceeds maximum_rgb_frame_bytes"));
}

#[test]
fn update_is_revisioned_noop_is_stable_and_stale_cas_is_rejected() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, config_path) = authority(&directory);
    let update = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"fps": 45}}}
        }),
    );
    assert_eq!(update["ok"], true);
    assert_eq!(update["revision"], 2);
    assert_eq!(update["updated_fields"], json!(["vision.camera.fps"]));
    assert!(config_path.exists());

    let no_op = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 2,
            "data": {"vision": {"camera": {"fps": 45}}}
        }),
    );
    assert_eq!(no_op["revision"], 2);
    assert_eq!(no_op["updated_fields"], json!([]));

    let conflict = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"fps": 46}}}
        }),
    );
    assert_eq!(conflict["error"]["code"], "revision_conflict");
    assert_eq!(conflict["current_revision"], 2);
    let persisted: Value = serde_json::from_slice(&fs::read(config_path).unwrap()).unwrap();
    assert_eq!(persisted["vision"]["camera"]["fps"], 45);
    assert_eq!(persisted["metadata"]["configuration_revision"], 2);
}

#[test]
fn full_merged_validation_rejects_cross_field_invalid_update_without_writing() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, config_path) = authority(&directory);
    let response = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"video_start_frame": 100, "video_end_frame": 50}}}
        }),
    );
    assert_eq!(response["error"]["code"], "validation_failed");
    assert!(response["validation_errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|error| error["field"] == "vision.camera.video_end_frame"));
    assert!(
        !config_path.exists(),
        "failed validation must leave disk unchanged"
    );
}

#[test]
fn update_rejects_non_openvino_live_inference_providers() {
    for provider in ["cpu", "auto", "coral"] {
        let directory = TempDir::new().expect("temporary directory");
        let (authority, config_path) = authority(&directory);
        let response = call(
            &authority,
            json!({
                "protocol_version": 1,
                "operation": "update",
                "expected_revision": 1,
                "data": {"perception": {"rust_live": {"provider": provider}}}
            }),
        );

        assert_eq!(response["error"]["code"], "validation_failed");
        assert!(response["validation_errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| error["field"] == "perception.rust_live.provider"));
        assert!(
            !config_path.exists(),
            "rejected provider updates must leave the active configuration unchanged"
        );
    }
}

#[test]
fn generic_expiry_must_outlast_the_configured_maximum_observation_age() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, config_path) = authority(&directory);
    let response = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {
                "perception": {
                    "freshness": {
                        "maximum_observation_age_seconds": 2.0,
                        "expire_after_ms": 2_000
                    },
                    "rust_live": { "provider": "openvino" }
                }
            }
        }),
    );
    assert_eq!(response["error"]["code"], "validation_failed");
    assert!(response["validation_errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(
            |error| error["field"] == "perception.freshness.expire_after_ms"
                && error["message"] == "Expiry must be greater than the maximum observation age"
        ));
    assert!(
        !config_path.exists(),
        "invalid freshness policy must not be persisted"
    );
}

#[test]
fn trusted_proxy_validation_matches_ip_network_address_and_cidr_forms() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, _) = authority(&directory);

    let valid = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "validate",
            "data": {"api": {"rate_limiting": {"trusted_proxies": [
                "192.0.2.129/24", "127.0.0.1", "2001:db8::1/64"
            ]}}}
        }),
    );
    assert_eq!(valid["valid"], true);

    let invalid = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "validate",
            "data": {"api": {"rate_limiting": {"trusted_proxies": ["nonsense"]}}}
        }),
    );
    assert_eq!(invalid["valid"], false);
    assert!(invalid["validation_errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|error| error["field"] == "api.rate_limiting.trusted_proxies[0]"));
}

#[test]
fn replace_normalizes_imported_revision_and_persists_replacement() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, _) = authority(&directory);
    let response = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "replace",
            "expected_revision": 1,
            "data": {
                "metadata": {"configuration_revision": 9000000, "version": "1.0.0"},
                "test": {"value": 123}
            }
        }),
    );
    assert_eq!(response["ok"], true);
    assert_eq!(response["revision"], 2);
    assert_eq!(
        response["configuration"]["metadata"]["configuration_revision"],
        2
    );
    assert_eq!(response["configuration"].get("vision"), None);
}

#[test]
fn warning_requires_force_and_section_reset_and_export_are_supported() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, _) = authority(&directory);
    let warning = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"fps": 90}}}
        }),
    );
    assert_eq!(warning["error"]["code"], "warnings_require_force");
    assert_eq!(warning["current_revision"], 1);

    let forced = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "force_update": true,
            "data": {"vision": {"camera": {"fps": 90}}}
        }),
    );
    assert_eq!(forced["revision"], 2);
    assert_eq!(forced["warnings"].as_array().unwrap().len(), 1);

    let reset = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "reset_section",
            "expected_revision": 2,
            "sections": ["vision", "metadata"]
        }),
    );
    assert_eq!(reset["revision"], 3);
    assert_eq!(reset["configuration"]["vision"]["camera"]["fps"], 30);
    assert_eq!(
        reset["configuration"]["metadata"]["configuration_revision"],
        3
    );
    assert!(!reset["updated_fields"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field == "metadata.configuration_revision"));

    let export = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "export",
            "sections": ["vision"],
            "include_metadata": false
        }),
    );
    assert_eq!(export["configuration"].as_object().unwrap().len(), 1);
    assert!(export["configuration"].get("metadata").is_none());
}

#[test]
fn malformed_and_oversized_requests_are_explicitly_rejected() {
    let directory = TempDir::new().expect("temporary directory");
    let (authority, _) = authority(&directory);
    let (malformed, ok) = authority.handle_json(b"{not-json");
    assert!(!ok);
    assert_eq!(
        serde_json::from_slice::<Value>(&malformed).unwrap()["error"]["code"],
        "invalid_request"
    );

    let limits = AuthorityLimits {
        max_input_bytes: 16,
        ..AuthorityLimits::default()
    };
    let limited =
        ConfigurationAuthority::new(directory.path().join("limited.json"), limits).unwrap();
    let (response, ok) = limited.handle_json(&[b' '; 17]);
    assert!(!ok);
    assert_eq!(
        serde_json::from_slice::<Value>(&response).unwrap()["error"]["code"],
        "request_too_large"
    );

    let small_output = AuthorityLimits {
        max_output_bytes: 256,
        ..AuthorityLimits::default()
    };
    let output_limited =
        ConfigurationAuthority::new(directory.path().join("output-limited.json"), small_output)
            .unwrap();
    let (response, ok) =
        output_limited.handle_json(&serde_json::to_vec(&request("schema")).unwrap());
    assert!(!ok);
    assert!(response.len() <= 256);
    assert_eq!(
        serde_json::from_slice::<Value>(&response).unwrap()["error"]["code"],
        "response_too_large"
    );
}

#[test]
fn oversized_mutation_response_does_not_commit_or_advance_revision() {
    let directory = TempDir::new().expect("temporary directory");
    let config_path = directory.path().join("output-limited-mutation.json");
    let limits = AuthorityLimits {
        max_output_bytes: 256,
        ..AuthorityLimits::default()
    };
    let authority = ConfigurationAuthority::new(&config_path, limits).unwrap();

    let (response, ok) = authority.handle_json(
        &serde_json::to_vec(&json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"fps": 45}}}
        }))
        .unwrap(),
    );

    assert!(!ok);
    let response: Value = serde_json::from_slice(&response).unwrap();
    assert_eq!(response["error"]["code"], "response_too_large");
    assert_eq!(response["operation"], "update");
    assert!(
        !config_path.exists(),
        "rejected output must not create the store"
    );
}

#[test]
fn oversized_legacy_snapshot_does_not_persist_revision_migration() {
    let directory = TempDir::new().expect("temporary directory");
    let (source, config_path) = authority(&directory);
    let mut legacy = call(&source, request("defaults"))["configuration"].clone();
    legacy["metadata"]
        .as_object_mut()
        .unwrap()
        .remove("configuration_revision");
    let legacy_bytes = serde_json::to_vec_pretty(&legacy).unwrap();
    fs::write(&config_path, &legacy_bytes).unwrap();

    let limits = AuthorityLimits {
        max_output_bytes: 256,
        ..AuthorityLimits::default()
    };
    let authority = ConfigurationAuthority::new(&config_path, limits).unwrap();
    let (response, ok) = authority.handle_json(&serde_json::to_vec(&request("snapshot")).unwrap());

    assert!(!ok);
    assert_eq!(
        serde_json::from_slice::<Value>(&response).unwrap()["error"]["code"],
        "response_too_large"
    );
    assert_eq!(
        fs::read(&config_path).unwrap(),
        legacy_bytes,
        "an undeliverable migration response must leave the legacy file unchanged"
    );
}

#[test]
fn malformed_and_oversized_store_files_fail_closed_without_rewriting() {
    let directory = TempDir::new().expect("temporary directory");
    let malformed_path = directory.path().join("malformed.json");
    let malformed_bytes = b"{not-json";
    fs::write(&malformed_path, malformed_bytes).unwrap();
    let malformed_authority =
        ConfigurationAuthority::new(&malformed_path, AuthorityLimits::default()).unwrap();
    let malformed = call(&malformed_authority, request("snapshot"));
    assert_eq!(malformed["error"]["code"], "store_error");
    assert_eq!(fs::read(&malformed_path).unwrap(), malformed_bytes);

    let oversized_path = directory.path().join("oversized.json");
    fs::write(&oversized_path, vec![b' '; 1025]).unwrap();
    let limits = AuthorityLimits {
        max_config_bytes: 1024,
        ..AuthorityLimits::default()
    };
    let oversized_authority = ConfigurationAuthority::new(&oversized_path, limits).unwrap();
    let oversized = call(&oversized_authority, request("snapshot"));
    assert_eq!(oversized["error"]["code"], "store_too_large");
    assert_eq!(fs::metadata(oversized_path).unwrap().len(), 1025);
}

#[test]
fn snapshots_reopen_from_disk_and_migrate_legacy_revision_atomically() {
    let directory = TempDir::new().expect("temporary directory");
    let path = directory.path().join("legacy.json");
    fs::write(
        &path,
        br#"{"metadata":{"version":"1.0.0"},"vision":{"camera":{"fps":42}}}"#,
    )
    .unwrap();
    let authority = ConfigurationAuthority::new(&path, AuthorityLimits::default()).unwrap();
    let snapshot = call(&authority, request("snapshot"));
    assert_eq!(snapshot["revision"], 1);
    assert_eq!(snapshot["migration"], "legacy_revision_added");
    let migrated: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(migrated["metadata"]["configuration_revision"], 1);
    assert_eq!(migrated["vision"]["camera"]["fps"], 42);

    let update = call(
        &authority,
        json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"fps": 44}}}
        }),
    );
    let restarted = ConfigurationAuthority::new(&path, AuthorityLimits::default()).unwrap();
    let readback = call(&restarted, request("snapshot"));
    assert_eq!(update["revision"], 2);
    assert_eq!(readback["revision"], 2);
    assert_eq!(readback["configuration"]["vision"]["camera"]["fps"], 44);
}

#[test]
fn concurrent_compare_and_swap_allows_only_one_writer() {
    let directory = TempDir::new().expect("temporary directory");
    let path = directory.path().join("concurrent.json");
    let first = ConfigurationAuthority::new(&path, AuthorityLimits::default()).unwrap();
    let second = ConfigurationAuthority::new(&path, AuthorityLimits::default()).unwrap();
    let thread_a = thread::spawn(move || {
        call(
            &first,
            json!({"protocol_version":1,"operation":"update","expected_revision":1,"data":{"vision":{"camera":{"fps":31}}}}),
        )
    });
    let thread_b = thread::spawn(move || {
        call(
            &second,
            json!({"protocol_version":1,"operation":"update","expected_revision":1,"data":{"vision":{"camera":{"fps":32}}}}),
        )
    });
    let results = [thread_a.join().unwrap(), thread_b.join().unwrap()];
    assert_eq!(
        results.iter().filter(|result| result["ok"] == true).count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| result["error"]["code"] == "revision_conflict")
            .count(),
        1
    );
}

#[test]
fn lock_wait_is_bounded_and_reports_timeout() {
    let directory = TempDir::new().expect("temporary directory");
    let path = directory.path().join("locked.json");
    let limits = AuthorityLimits {
        lock_timeout: Duration::from_millis(30),
        ..AuthorityLimits::default()
    };
    let authority = ConfigurationAuthority::new(&path, limits).unwrap();
    let lock_path = directory.path().join(".locked.json.lock");
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .unwrap();
    lock.lock_exclusive().unwrap();
    let response = call(&authority, request("snapshot"));
    FileExt::unlock(&lock).unwrap();
    assert_eq!(response["error"]["code"], "lock_timeout");
}

#[test]
fn command_binary_is_one_shot_bounded_json_and_requires_explicit_config_path() {
    let directory = TempDir::new().expect("temporary directory");
    let config_path = directory.path().join("cli-config.json");
    let mut child = Command::new(env!("CARGO_BIN_EXE_billiards-perception-config"))
        .arg("--config")
        .arg(&config_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start config binary");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"protocol_version":1,"operation":"snapshot"}"#)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap()["revision"],
        1
    );
    assert!(!config_path.exists());

    let missing_path = Command::new(env!("CARGO_BIN_EXE_billiards-perception-config"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(!missing_path.status.success());
}

#[test]
fn independent_command_processes_share_file_lock_and_revision_cas() {
    let directory = TempDir::new().expect("temporary directory");
    let config_path = directory.path().join("process-cas.json");
    let binary = env!("CARGO_BIN_EXE_billiards-perception-config");
    let mut children = Vec::new();
    for fps in [33, 34] {
        let mut child = Command::new(binary)
            .arg("--config")
            .arg(&config_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn config process");
        let body = serde_json::to_vec(&json!({
            "protocol_version": 1,
            "operation": "update",
            "expected_revision": 1,
            "data": {"vision": {"camera": {"fps": fps}}}
        }))
        .unwrap();
        child.stdin.take().unwrap().write_all(&body).unwrap();
        children.push(child);
    }
    let results: Vec<Value> = children
        .into_iter()
        .map(|child| {
            let output = child.wait_with_output().unwrap();
            serde_json::from_slice(&output.stdout).unwrap()
        })
        .collect();
    assert_eq!(
        results.iter().filter(|result| result["ok"] == true).count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| result["error"]["code"] == "revision_conflict")
            .count(),
        1
    );
}

#[test]
fn repository_live_config_is_never_the_implicit_test_target() {
    assert_eq!(DEFAULT_MAX_INPUT_BYTES, 1_048_576);
    assert_eq!(DEFAULT_MAX_OUTPUT_BYTES, 2_097_152);
    assert_eq!(DEFAULT_MAX_CONFIG_BYTES, 1_048_576);
    let directory = TempDir::new().expect("temporary directory");
    let (_, path) = authority(&directory);
    assert!(path.starts_with(directory.path()));
    assert!(!path.ends_with("config.json"));
}
