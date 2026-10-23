use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/server-audit-summary.json")).unwrap()
}

#[test]
fn command_summarizes_jsonl_using_the_reference_fixture() {
    let fixture = fixture();
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-server-summary"
    ))
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    {
        let input = child.stdin.as_mut().unwrap();
        for event in fixture["events"].as_array().unwrap() {
            serde_json::to_writer(&mut *input, event).unwrap();
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
fn command_rejects_event_count_and_line_size_overflows() {
    let event_limit = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-server-summary"
    ))
    .args(["--maximum-events", "1"])
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    let mut event_limit = event_limit;
    event_limit
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{}\n{}\n")
        .unwrap();
    let event_result = event_limit.wait_with_output().unwrap();
    assert!(!event_result.status.success());
    assert!(String::from_utf8_lossy(&event_result.stderr).contains("maximum event count"));

    let line_limit = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-server-summary"
    ))
    .args(["--maximum-line-bytes", "2"])
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    let mut line_limit = line_limit;
    line_limit
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{}\n")
        .unwrap();
    let line_result = line_limit.wait_with_output().unwrap();
    assert!(!line_result.status.success());
    assert!(String::from_utf8_lossy(&line_result.stderr).contains("exceeds 2 bytes"));

    let input_limit = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-server-summary"
    ))
    .args(["--maximum-input-bytes", "2"])
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    let mut input_limit = input_limit;
    input_limit
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{}\n")
        .unwrap();
    let input_result = input_limit.wait_with_output().unwrap();
    assert!(!input_result.status.success());
    assert!(String::from_utf8_lossy(&input_result.stderr).contains("maximum byte count"));
}
