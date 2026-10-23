use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use serde_json::Value;

#[test]
fn helper_uses_bounded_control_messages_and_reports_process_samples() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_billiards-performance-audit-resources"))
        .args([
            "--process-id",
            &std::process::id().to_string(),
            "--sample-interval-ms",
            "200",
            "--maximum-samples",
            "1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("resource helper should start");
    let mut input = child.stdin.take().expect("child stdin should be piped");
    let output = child.stdout.take().expect("child stdout should be piped");
    let mut output = BufReader::new(output);
    let mut line = String::new();

    output
        .read_line(&mut line)
        .expect("ready message should arrive");
    let ready: Value = serde_json::from_str(&line).expect("ready message should be JSON");
    assert_eq!(ready["kind"], "ready");
    assert_eq!(ready["schema_version"], 2);
    line.clear();

    input
        .write_all(b"{\"kind\":\"start\"}\n")
        .expect("start control should be written");
    input.flush().expect("start control should be flushed");
    output
        .read_line(&mut line)
        .expect("started message should arrive");
    let started: Value = serde_json::from_str(&line).expect("started message should be JSON");
    assert_eq!(started["kind"], "started");
    assert_eq!(started["schema_version"], 2);
    assert_eq!(started["process_id"], std::process::id());
    assert!(started["process_start_time_unix_seconds"].is_u64());
    line.clear();

    thread::sleep(Duration::from_millis(450));
    input
        .write_all(b"{\"kind\":\"stop\"}\n")
        .expect("stop control should be written");
    input.flush().expect("stop control should be flushed");
    output
        .read_line(&mut line)
        .expect("summary message should arrive");
    let summary_message: Value =
        serde_json::from_str(&line).expect("summary message should be JSON");
    drop(input);

    let status = child.wait().expect("resource helper should exit");
    assert!(status.success());
    assert_eq!(summary_message["kind"], "summary");
    assert_eq!(summary_message["summary"]["status"], "available");
    assert_eq!(summary_message["summary"]["process_id"], std::process::id());
    assert_eq!(
        summary_message["summary"]["statistics_window"],
        "retained_tail"
    );
    assert_eq!(
        summary_message["summary"]["process_identity_resolution"],
        "pid_plus_sysinfo_start_time_unix_seconds"
    );
    assert!(
        summary_message["summary"]["process_identity_resolution_note"]
            .as_str()
            .expect("identity resolution note should be text")
            .contains("one-second resolution")
    );
    assert!(
        summary_message["summary"]["sample_count"]
            .as_u64()
            .expect("sample count should be an integer")
            > 0
    );
    assert!(
        summary_message["summary"]["resident_memory_bytes"]["maximum"]
            .as_u64()
            .expect("RSS maximum should be an integer")
            > 0
    );
}

#[test]
fn helper_rejects_sample_capacity_that_can_exceed_the_summary_bound() {
    let output = Command::new(env!("CARGO_BIN_EXE_billiards-performance-audit-resources"))
        .args([
            "--process-id",
            &std::process::id().to_string(),
            "--sample-interval-ms",
            "200",
            "--maximum-samples",
            "50001",
        ])
        .output()
        .expect("resource helper should launch");

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("between 1 and 50000"));
}
