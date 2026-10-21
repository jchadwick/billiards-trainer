#![cfg(feature = "camera-gstreamer-discovery")]

use std::process::Command;

use serde_json::Value;

#[test]
fn discovery_command_emits_bounded_inventory_json_without_device_assumptions() {
    let maximum_bytes = 4_096;
    let maximum_bytes_argument = maximum_bytes.to_string();
    let output = Command::new(env!("CARGO_BIN_EXE_billiards-perception-camera-discovery"))
        .args(["--max-output-bytes", &maximum_bytes_argument])
        .output()
        .expect("camera discovery executable should start");

    assert!(
        output.status.success(),
        "camera discovery exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() <= maximum_bytes);

    let report: Value = serde_json::from_slice(&output.stdout)
        .expect("camera discovery should emit one valid JSON report");
    assert_eq!(report["schema_version"], 2);
    assert!(report["platform"]["os"].is_string());
    assert!(report["platform"]["architecture"].is_string());
    assert!(report["gstreamer_version"].is_string());
    assert!(
        report["provider_status"] == "started"
            || report["provider_status"] == "no_matching_provider"
    );
    assert!(report["devices"].is_array());
    assert!(report["truncated"].is_boolean());
    assert!(report["omitted_device_count"].is_number());

    for device in report["devices"].as_array().unwrap() {
        assert!(device["identifier_classification"].is_string());
        assert!(device["capture_selector"].is_string() || device["capture_selector"].is_null());
        assert!(device["metadata_omitted"].is_boolean());
        assert!(device["omitted_property_count"].is_number());
        assert!(device["advertised_caps_omitted"].is_boolean());
        assert!(device["display_name"].is_string() || device["display_name"].is_null());
        assert!(device["device_class"].is_string() || device["device_class"].is_null());
        assert!(device["advertised_caps"].is_string() || device["advertised_caps"].is_null());
        assert!(
            device["provider_properties"].is_object() || device["provider_properties"].is_null()
        );
    }
}
