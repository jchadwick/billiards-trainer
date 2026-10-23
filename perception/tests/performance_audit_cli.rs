//! Process-level smoke coverage for the canonical live performance audit CLI.
//!
//! The fixture deliberately speaks only the anonymous state WebSocket
//! contract. This verifies the compiled executable and its artifact workflow
//! without a camera, model, or application server.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::time::{interval, timeout, Instant, MissedTickBehavior};
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn state_message(sequence: u64) -> String {
    serde_json::json!({
        "type": "state",
        "sequence": sequence,
        "data": {
            "sequence": sequence,
            "balls": [],
            "observation_status": {
                "engine_instance_id": "performance-audit-cli-smoke",
                "source_session_id": "fixture-session",
                "source_frame_id": format!("fixture-frame-{sequence}"),
                "source_frame_number": sequence,
                "freshness_status": "fresh",
                "ball_positions_status": "observed",
                "component_observation_status": {"balls": "observed"}
            }
        }
    })
    .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compiled_cli_receives_fresh_states_and_writes_json_and_markdown_artifacts() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let fixture = tokio::spawn(async move {
        let (stream, _) = timeout(Duration::from_secs(5), listener.accept())
            .await
            .expect("audit CLI did not connect to the local state fixture")
            .unwrap();
        let socket = accept_async(stream).await.unwrap();
        let (mut sender, mut receiver) = socket.split();
        let mut frames = interval(Duration::from_millis(20));
        frames.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let stop = Instant::now() + Duration::from_secs(8);
        let mut sequence = 0_u64;

        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(stop) => break,
                _ = frames.tick() => {
                    sequence += 1;
                    if sender.send(Message::Text(state_message(sequence).into())).await.is_err() {
                        break;
                    }
                }
                incoming = receiver.next() => match incoming {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(payload))) => {
                        if sender.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Some(Err(error)) => panic!("state fixture WebSocket failed: {error}"),
                    Some(Ok(_)) => {}
                }
            }
        }
    });

    let temporary = tempdir().unwrap();
    let output_root = temporary.path().join("audits");
    let mut command = Command::new(env!("CARGO_BIN_EXE_billiards-performance-audit-run"));
    command
        .arg("--url")
        .arg(format!("ws://{address}/state"))
        .arg("--warmup-seconds")
        .arg("0")
        .arg("--duration-seconds")
        .arg("1")
        .arg("--repetitions")
        .arg("1")
        .arg("--connect-timeout-seconds")
        .arg("2")
        .arg("--summary-interval-seconds")
        .arg("1")
        .arg("--maximum-samples")
        .arg("4")
        .arg("--output-root")
        .arg(&output_root)
        .kill_on_drop(true);

    let process_output = match timeout(Duration::from_secs(20), command.output()).await {
        Ok(output) => output.expect("could not launch performance audit CLI"),
        Err(_) => {
            fixture.abort();
            panic!("performance audit CLI exceeded its bounded 20-second deadline");
        }
    };
    if !process_output.status.success() {
        fixture.abort();
        panic!(
            "performance audit CLI failed with {}\nstdout:\n{}\nstderr:\n{}",
            process_output.status,
            String::from_utf8_lossy(&process_output.stdout),
            String::from_utf8_lossy(&process_output.stderr),
        );
    }

    timeout(Duration::from_secs(2), fixture)
        .await
        .expect("local state fixture did not shut down after the audit")
        .expect("local state fixture task panicked");

    let stdout = String::from_utf8(process_output.stdout).unwrap();
    let artifact_line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("audit artifacts: "))
        .expect("CLI did not report its artifact directory");
    let run_directory = std::path::PathBuf::from(artifact_line);
    assert!(run_directory.starts_with(&output_root));

    let summary: Value =
        serde_json::from_slice(&std::fs::read(run_directory.join("summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["tool"], "billiards-performance-audit-run");
    assert_eq!(summary["mode"], "repeated");
    assert_eq!(summary["configuration"]["maximum_samples_per_run"], 4);
    assert_eq!(summary["measured_runs"].as_array().unwrap().len(), 1);

    let run = &summary["measured_runs"][0];
    let report_path = run_directory.join(run["report_path"].as_str().unwrap());
    let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
    let counts = &report["report"]["counts"];
    let freshness = &report["report"]["freshness"];
    let completeness = &report["report"]["completeness"];
    assert!(counts["state_messages"].as_u64().unwrap() >= 2);
    assert_eq!(counts["invalid_json_messages"], 0);
    assert_eq!(
        freshness["unique_observed_source_observations"],
        counts["state_messages"]
    );
    assert_eq!(completeness["incomplete_state_messages"], 0);
    assert!(
        report["report"]["timing"]["local_state_receive_interval_ms"]["sample_count"]
            .as_u64()
            .unwrap()
            >= 1
    );

    let details_path = run_directory.join(run["details_path"].as_str().unwrap());
    let details = std::fs::read_to_string(details_path).unwrap();
    assert_eq!(
        details.lines().count(),
        4,
        "details must honor the sample cap"
    );
    let markdown = std::fs::read_to_string(run_directory.join("report.md")).unwrap();
    assert!(markdown.contains("State delivery audit"));
    assert!(markdown.contains("Observed source frames"));
}
