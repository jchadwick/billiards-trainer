use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/overhead-comparison.json")).unwrap()
}

#[test]
fn command_matches_the_shared_overhead_comparison_fixture() {
    let fixture = fixture();
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-overhead-summary"
    ))
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    serde_json::to_writer(child.stdin.as_mut().unwrap(), &fixture["input"]).unwrap();
    drop(child.stdin.take());
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
fn command_rejects_oversized_and_malformed_comparison_documents() {
    let oversized = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-overhead-summary"
    ))
    .args(["--maximum-input-bytes", "2"])
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    let mut oversized = oversized;
    oversized
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"{}\n")
        .unwrap();
    let oversized = oversized.wait_with_output().unwrap();
    assert!(!oversized.status.success());
    assert!(String::from_utf8_lossy(&oversized.stderr).contains("maximum byte count 2"));

    let mut malformed = fixture()["input"].clone();
    malformed["minimum_repetitions"] = Value::from(0);
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-performance-audit-overhead-summary"
    ))
    .stdin(Stdio::piped())
    .stdout(Stdio::null())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    serde_json::to_writer(child.stdin.as_mut().unwrap(), &malformed).unwrap();
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("minimum_repetitions"));
}
