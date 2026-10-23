use std::time::Duration;
use std::{fs, process::Stdio};

use billiards_perception::performance_audit_receiver::{connect_and_collect, StateReceiverConfig};
use futures_util::SinkExt;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn state_message(sequence: u64, frame_id: &str) -> String {
    json!({
        "type": "state",
        "sequence": sequence,
        "data": {
            "sequence": sequence,
            "balls": [{"id": "cue"}, {"id": "object"}],
            "observation_status": {
                "freshness_status": "fresh",
                "ball_positions_status": "observed",
                "source_timestamp_kind": "monotonic:engine-a:camera_read_complete",
                "frame_publication_age_seconds": 0.02,
                "observation_identity": {
                    "engine_instance_id": "engine-a",
                    "session_id": "session-a",
                    "frame_id": frame_id
                }
            }
        }
    })
    .to_string()
}

#[tokio::test]
async fn receives_live_websocket_states_and_reports_gaps_and_freshness() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        socket
            .send(Message::Text(state_message(10, "100").into()))
            .await
            .unwrap();
        socket
            .send(Message::Text(state_message(12, "102").into()))
            .await
            .unwrap();
        socket.close(None).await.unwrap();
    });

    let config = StateReceiverConfig {
        duration: Duration::from_secs(2),
        connect_timeout: Duration::from_secs(1),
        summary_interval: Duration::from_secs(1),
        maximum_message_bytes: 4096,
        maximum_samples: 16,
        resource_sample_process_id: None,
        resource_sample_interval: Duration::from_secs(1),
    };
    let output = connect_and_collect(&format!("ws://{address}/state"), &config)
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(output.report.termination_reason, "peer_closed");
    assert_eq!(output.report.counts.state_messages, 2);
    assert_eq!(output.report.freshness.fresh_state_messages, 2);
    assert_eq!(output.report.freshness.fresh_unique_source_observations, 2);
    assert_eq!(output.report.stream_order.sequence_gaps, 1);
    assert_eq!(output.report.stream_order.missing_sequence_total, 1);
    assert_eq!(
        output.report.source_observations.source_frame_sequence_gaps,
        1
    );
    assert_eq!(
        output
            .report
            .source_observations
            .missing_source_frame_sequence_total,
        1
    );
    assert_eq!(
        output
            .report
            .timing
            .read_completion_to_publication_age_ms
            .sample_count,
        2
    );
    assert_eq!(output.retained_samples.len(), 2);
    assert_eq!(output.retained_samples[1].ball_count, Some(2));
    assert_eq!(
        output.report.cross_process_source_to_receipt_latency.status,
        "unavailable_unverified_clock"
    );
    assert!(output.report.process_resources.is_none());
}

#[tokio::test]
async fn command_exports_versioned_json_and_bounded_jsonl_details() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        socket
            .send(Message::Text(state_message(1, "42").into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(480)).await;
        socket.close(None).await.unwrap();
    });

    let directory = tempfile::tempdir().unwrap();
    let report_path = directory.path().join("audit.json");
    let details_path = directory.path().join("states.jsonl");
    let result = Command::new(env!("CARGO_BIN_EXE_billiards-performance-audit"))
        .args([
            "--url",
            &format!("ws://{address}/state"),
            "--duration-seconds",
            "2",
            "--output-json",
            report_path.to_str().unwrap(),
            "--details-jsonl",
            details_path.to_str().unwrap(),
            "--resource-process-id",
            &std::process::id().to_string(),
            "--resource-sample-interval-ms",
            "200",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .unwrap();
    server.await.unwrap();

    assert!(
        result.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("audit complete"));

    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["report"]["schema_version"], 2);
    assert_eq!(report["report"]["counts"]["state_messages"], 1);
    assert_eq!(report["retained_samples"].as_array().unwrap().len(), 1);
    assert_eq!(report["report"]["process_resources"]["status"], "available");
    assert!(
        report["report"]["process_resources"]["sample_count"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        report["report"]["process_resources"]["resident_memory_bytes"]["maximum"]
            .as_u64()
            .unwrap()
            > 0
    );

    let details = fs::read_to_string(details_path).unwrap();
    let samples: Vec<serde_json::Value> = details
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0]["source_frame_id"], "42");
}
