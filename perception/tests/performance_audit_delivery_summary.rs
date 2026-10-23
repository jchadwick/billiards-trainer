use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/delivery-latency-summary.json")).unwrap()
}

#[test]
fn command_matches_the_shared_delivery_latency_fixture() {
    let fixture = fixture();
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-delivery-summary"
    ))
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    {
        let input = child.stdin.as_mut().unwrap();
        for record in fixture["records"].as_array().unwrap() {
            serde_json::to_writer(&mut *input, record).unwrap();
            input.write_all(b"\n").unwrap();
        }
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "summary command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary, fixture["expected"]);
}

#[test]
fn command_requires_configuration_and_enforces_event_bounds() {
    let mut missing_config = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-delivery-summary"
    ))
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    missing_config
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{\"kind\":\"client\",\"event\":{}}\n")
        .unwrap();
    let missing_config = missing_config.wait_with_output().unwrap();
    assert!(!missing_config.status.success());
    assert!(
        String::from_utf8_lossy(&missing_config.stderr).contains("first record must be the config")
    );

    let fixture = fixture();
    let records = fixture["records"].as_array().unwrap();
    let mut bounded = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-delivery-summary"
    ))
    .args(["--maximum-events", "1"])
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    {
        let input = bounded.stdin.as_mut().unwrap();
        for record in records {
            serde_json::to_writer(&mut *input, record).unwrap();
            input.write_all(b"\n").unwrap();
        }
    }
    let bounded = bounded.wait_with_output().unwrap();
    assert!(!bounded.status.success());
    assert!(String::from_utf8_lossy(&bounded.stderr).contains("maximum event count 1"));
}

#[test]
fn command_rejects_invalid_configuration_values() {
    let invalid_config = br#"{"kind":"config","deadline_budget_ms":-1,"clock_name":"clock","server_clock_domain_id":null}"#;
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-delivery-summary"
    ))
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(invalid_config)
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("finite and non-negative"));
}
