use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/trajectory-summary.json")).unwrap()
}

fn run_records(arguments: &[&str], records: &[Value]) -> std::process::Output {
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-trajectory-summary"
    ))
    .args(arguments)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    {
        let input = child.stdin.as_mut().unwrap();
        for record in records {
            serde_json::to_writer(&mut *input, record).unwrap();
            input.write_all(b"\n").unwrap();
        }
    }
    child.wait_with_output().unwrap()
}

#[test]
fn streaming_command_matches_shared_trajectory_fixture() {
    let fixture = fixture();
    let output = run_records(&[], fixture["records"].as_array().unwrap());
    assert!(
        output.status.success(),
        "summary command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary, fixture["expected"]);
}

#[test]
fn command_rejects_malformed_protocol_missing_values_and_boolean_metrics() {
    let cases = [
        (
            vec![serde_json::json!({"type":"trajectory_receipt","event":{}})],
            "first record must be the config",
        ),
        (
            vec![serde_json::json!({"type":"config","maximum_samples":1})],
            "finish record",
        ),
        (
            vec![
                serde_json::json!({"type":"config","maximum_samples":1}),
                serde_json::json!({"type":"trajectory_receipt","event":{"stream_sequence":true}}),
                serde_json::json!({"type":"finish","duration_seconds":1.0}),
            ],
            "stream_sequence must be a non-negative integer",
        ),
        (
            vec![
                serde_json::json!({"type":"config","maximum_samples":1}),
                serde_json::json!({"type":"trajectory_receipt","event":{"source_publication_to_client_receipt_ms":true}}),
                serde_json::json!({"type":"finish","duration_seconds":1.0}),
            ],
            "source_publication_to_client_receipt_ms must be a finite number",
        ),
        (
            vec![
                serde_json::json!({"type":"config","maximum_samples":1}),
                serde_json::json!({"type":"finish","duration_seconds":1.0}),
                serde_json::json!({"type":"trajectory_receipt","event":{}}),
            ],
            "finish record must be the final",
        ),
    ];
    for (records, expected_error) in cases {
        let output = run_records(&[], &records);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected_error),
            "expected {expected_error:?}, got {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn command_enforces_configured_samples_line_record_and_output_limits() {
    let fixture = fixture();
    let records = fixture["records"].as_array().unwrap();

    let sample_limit = run_records(&["--maximum-samples", "1"], records);
    assert!(!sample_limit.status.success());
    assert!(String::from_utf8_lossy(&sample_limit.stderr).contains("exceeds CLI limit 1"));

    let record_limit = run_records(&["--maximum-records", "2"], records);
    assert!(!record_limit.status.success());
    assert!(String::from_utf8_lossy(&record_limit.stderr).contains("maximum record count 2"));

    let input_limit = run_records(
        &["--maximum-input-bytes", "60"],
        &[
            serde_json::json!({"type":"config","maximum_samples":1}),
            serde_json::json!({"type":"finish","duration_seconds":1.0}),
        ],
    );
    assert!(!input_limit.status.success());
    assert!(String::from_utf8_lossy(&input_limit.stderr).contains("maximum byte count 60"));

    let output_limit = run_records(&["--maximum-output-bytes", "1"], records);
    assert!(!output_limit.status.success());
    assert!(String::from_utf8_lossy(&output_limit.stderr).contains("summary output exceeds"));

    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-trajectory-summary"
    ))
    .args(["--maximum-line-bytes", "20"])
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{\"type\":\"config\",\"maximum_samples\":1}\n")
        .unwrap();
    let line_limit = child.wait_with_output().unwrap();
    assert!(!line_limit.status.success());
    assert!(String::from_utf8_lossy(&line_limit.stderr).contains("exceeds 20 bytes"));
}

#[test]
fn command_rejects_malformed_json_lines() {
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-trajectory-summary"
    ))
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{not-json}\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid JSON record"));
    assert!(output.stdout.is_empty());
}

#[test]
fn command_rejects_invalid_config_and_finish_metadata() {
    for (config, finish, expected_error) in [
        (
            serde_json::json!({"type":"config","maximum_samples":true}),
            serde_json::json!({"type":"finish","duration_seconds":1.0}),
            "maximum_samples must be a positive integer",
        ),
        (
            serde_json::json!({"type":"config","maximum_samples":1}),
            serde_json::json!({"type":"finish","duration_seconds":true}),
            "duration_seconds must be a finite non-negative number",
        ),
        (
            serde_json::json!({"type":"config","maximum_samples":1}),
            serde_json::json!({"type":"finish","duration_seconds":1.0,"bridge_loss_count":true}),
            "bridge_loss_count must be a non-negative integer",
        ),
    ] {
        let output = run_records(&[], &[config, finish]);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected_error));
    }
}
