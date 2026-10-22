use std::net::SocketAddr;
use std::time::Duration;

use billiards_perception::detection_batch_receiver::{receive_stream, serve_once, ReceiverConfig};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, DuplexStream};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::sync::oneshot;

const SUCCESS: &str = include_str!("../contracts/detection-batch/v1/detection-batch-success.json");
const EMPTY_SUCCESS: &str =
    include_str!("../contracts/detection-batch/v1/detection-batch-empty-success.json");

fn config(expected_batches: usize) -> ReceiverConfig {
    ReceiverConfig {
        expected_batches,
        maximum_batches: 10,
        maximum_message_bytes: 64 * 1024,
        accept_timeout: Duration::from_secs(1),
        session_timeout: Duration::from_secs(2),
        read_timeout: Duration::from_secs(1),
        maximum_detail_records: 10,
    }
}

fn framed(values: &[Value]) -> Vec<u8> {
    let mut output = Vec::new();
    for value in values {
        let payload = serde_json::to_vec(value).unwrap();
        let length = u32::try_from(payload.len()).unwrap();
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(&payload);
    }
    output
}

async fn receive_bytes(
    input: &[u8],
    receiver_config: ReceiverConfig,
) -> (
    billiards_perception::detection_batch_receiver::ReceiverReport,
    Vec<u8>,
) {
    let (mut producer, mut reader) = tokio::io::duplex(input.len().max(1) + 1);
    producer.write_all(input).await.unwrap();
    producer.shutdown().await.unwrap();
    receive_from_duplex(&mut reader, receiver_config).await
}

async fn receive_from_duplex(
    reader: &mut DuplexStream,
    receiver_config: ReceiverConfig,
) -> (
    billiards_perception::detection_batch_receiver::ReceiverReport,
    Vec<u8>,
) {
    let mut jsonl = Vec::new();
    let report = receive_stream(reader, &mut jsonl, &receiver_config)
        .await
        .unwrap();
    (report, jsonl)
}

#[tokio::test]
async fn accepts_ordered_batches_and_preserves_original_json_payloads() {
    let first: Value = serde_json::from_str(SUCCESS).unwrap();
    let second: Value = serde_json::from_str(EMPTY_SUCCESS).unwrap();
    let input = framed(&[first, second]);

    let (report, jsonl) = receive_bytes(&input, config(2)).await;

    assert!(report.complete);
    assert!(report.ordered);
    assert_eq!(report.received_batches, 2);
    assert_eq!(report.missing_sequences, 0);
    assert_eq!(report.duplicate_sequences, 0);
    assert!(report
        .cross_process_one_way_latency
        .starts_with("unavailable"));
    assert!(report.receipt_records[0]
        .receipt_monotonic_ns
        .parse::<u64>()
        .is_ok());
    let lines: Vec<_> = String::from_utf8(jsonl)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        serde_json::from_str::<Value>(&lines[1]).unwrap()["batch_sequence"],
        2
    );
}

#[tokio::test]
async fn reports_duplicate_missing_gap_and_reordered_sequences() {
    let first: Value = serde_json::from_str(SUCCESS).unwrap();
    let mut duplicate = first.clone();
    duplicate["source"]["frame_id"] = json!("frame-1043");

    let (report, _) = receive_bytes(&framed(&[first.clone(), duplicate]), config(2)).await;
    assert!(report.complete);
    assert!(!report.ordered);
    assert_eq!(report.duplicate_sequences, 1);
    assert_eq!(report.missing_sequences, 1);
    assert_eq!(report.out_of_order_records, 1);

    let mut third: Value = serde_json::from_str(EMPTY_SUCCESS).unwrap();
    third["batch_sequence"] = json!(3);
    third["source"]["frame_id"] = json!("frame-1044");
    let second: Value = serde_json::from_str(EMPTY_SUCCESS).unwrap();
    let (report, _) = receive_bytes(&framed(&[first, third, second]), config(3)).await;
    assert!(!report.ordered);
    assert_eq!(report.duplicate_sequences, 0);
    assert_eq!(report.missing_sequences, 0);
    assert_eq!(report.sequence_gap_events, 1);
    assert_eq!(report.sequence_gap_total, 1);
    assert_eq!(report.out_of_order_records, 1);
}

#[tokio::test]
async fn rejects_producer_identity_changes_and_schema_invalid_batches() {
    let first: Value = serde_json::from_str(SUCCESS).unwrap();
    let mut changed = serde_json::from_str::<Value>(EMPTY_SUCCESS).unwrap();
    changed["engine_instance_id"] = json!("different-engine");

    let (report, jsonl) = receive_bytes(&framed(&[first, changed]), config(2)).await;
    assert!(!report.complete);
    assert_eq!(report.received_batches, 1);
    assert!(report.errors[0].contains("identity changed"));
    assert_eq!(String::from_utf8(jsonl).unwrap().lines().count(), 1);

    let mut invalid: Value = serde_json::from_str(SUCCESS).unwrap();
    invalid["unexpected"] = json!(true);
    let (report, jsonl) = receive_bytes(&framed(&[invalid]), config(1)).await;
    assert!(!report.complete);
    assert_eq!(report.received_batches, 0);
    assert!(report.errors[0].starts_with("invalid_batch:"));
    assert!(jsonl.is_empty());
}

#[tokio::test]
async fn rejects_zero_oversized_and_truncated_frames() {
    let mut receiver_config = config(1);
    receiver_config.maximum_message_bytes = 16;
    let (report, _) = receive_bytes(&0u32.to_be_bytes(), receiver_config.clone()).await;
    assert_eq!(report.errors, ["zero_length_frame"]);

    let oversized = 17u32.to_be_bytes();
    let (report, _) = receive_bytes(&oversized, receiver_config.clone()).await;
    assert_eq!(report.errors, ["frame_exceeds_configured_maximum"]);

    let mut truncated = 3u32.to_be_bytes().to_vec();
    truncated.extend_from_slice(b"{}");
    let (report, _) = receive_bytes(&truncated, receiver_config).await;
    assert_eq!(report.errors, ["truncated_frame"]);
}

#[tokio::test]
async fn caps_receipt_details_and_sequence_values() {
    let first: Value = serde_json::from_str(SUCCESS).unwrap();
    let second: Value = serde_json::from_str(EMPTY_SUCCESS).unwrap();
    let mut receiver_config = config(2);
    receiver_config.maximum_detail_records = 1;

    let (report, _) = receive_bytes(&framed(&[first, second]), receiver_config).await;

    assert_eq!(report.received_batches, 2);
    assert_eq!(report.receipt_records_retained, 1);
    assert_eq!(report.receipt_records_dropped, 1);
    assert_eq!(report.sequence_values, None);
}

#[tokio::test]
async fn reports_read_timeout_without_hanging_on_a_slow_sender() {
    let (_producer, mut reader) = tokio::io::duplex(32);
    let mut receiver_config = config(1);
    receiver_config.read_timeout = Duration::from_millis(10);

    let (report, jsonl) = receive_from_duplex(&mut reader, receiver_config).await;

    assert_eq!(report.errors, ["read_timeout"]);
    assert!(!report.complete);
    assert!(jsonl.is_empty());
}

#[tokio::test]
async fn loopback_server_accepts_and_reports_one_real_tcp_session() {
    let (ready_sender, ready_receiver) = oneshot::channel::<SocketAddr>();
    let receiver_task = tokio::spawn(async move {
        let mut jsonl = Vec::new();
        serve_once(
            "127.0.0.1:0".parse().unwrap(),
            config(1),
            &mut jsonl,
            move |address| ready_sender.send(address).unwrap(),
        )
        .await
        .map(|(report, bound)| (report, jsonl, bound))
    });
    let address = ready_receiver.await.unwrap();
    let mut producer = TcpStream::connect(address).await.unwrap();
    producer
        .write_all(&framed(&[serde_json::from_str(SUCCESS).unwrap()]))
        .await
        .unwrap();
    producer.shutdown().await.unwrap();

    let (report, jsonl, bound) = receiver_task.await.unwrap().unwrap();
    assert!(report.complete);
    assert_eq!(report.peer_address.as_ref().unwrap().0, "127.0.0.1");
    assert_eq!(bound.ip().to_string(), "127.0.0.1");
    assert_eq!(
        serde_json::from_slice::<Value>(&jsonl).unwrap()["batch_sequence"],
        1
    );
}

#[tokio::test]
async fn receiver_refuses_non_loopback_bind_addresses() {
    let mut jsonl = Vec::new();
    let result = serve_once(
        "192.0.2.1:0".parse().unwrap(),
        config(1),
        &mut jsonl,
        |_| panic!("receiver must reject before announcing readiness"),
    )
    .await;

    assert!(result.unwrap_err().to_string().contains("loopback"));
}

#[tokio::test]
async fn command_writes_versioned_report_and_hashes_jsonl_without_overwriting() {
    let directory = tempfile::tempdir().unwrap();
    let jsonl_path = directory.path().join("received.jsonl");
    let report_path = directory.path().join("receiver.json");
    let mut child = Command::new(env!(
        "CARGO_BIN_EXE_billiards-perception-receive-detection-batches"
    ))
    .args([
        "--bind-address",
        "127.0.0.1:0",
        "--expected-batches",
        "1",
        "--maximum-batches",
        "10",
        "--maximum-message-bytes",
        "65536",
        "--accept-timeout-ms",
        "2000",
        "--session-timeout-ms",
        "2000",
        "--read-timeout-ms",
        "1000",
        "--batches-jsonl",
    ])
    .arg(&jsonl_path)
    .arg("--output-json")
    .arg(&report_path)
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::piped())
    .spawn()
    .unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut stdout = BufReader::new(stdout);
    let mut ready_line = String::new();
    stdout.read_line(&mut ready_line).await.unwrap();
    let ready: Value = serde_json::from_str(&ready_line).unwrap();
    assert_eq!(ready["event"], "ready");
    let address: SocketAddr = ready["address"].as_str().unwrap().parse().unwrap();

    let mut producer = TcpStream::connect(address).await.unwrap();
    producer
        .write_all(&framed(&[serde_json::from_str(SUCCESS).unwrap()]))
        .await
        .unwrap();
    producer.shutdown().await.unwrap();

    let mut report_stdout = String::new();
    stdout.read_to_string(&mut report_stdout).await.unwrap();
    let status = child.wait().await.unwrap();
    assert!(status.success());
    let report: Value = serde_json::from_str(&report_stdout).unwrap();
    let file_report: Value =
        serde_json::from_slice(&tokio::fs::read(&report_path).await.unwrap()).unwrap();
    let jsonl = tokio::fs::read(&jsonl_path).await.unwrap();
    assert_eq!(report, file_report);
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["complete"], true);
    assert_eq!(report["runtime"]["language"], "rust");
    assert_eq!(
        report["batches_jsonl_path"],
        jsonl_path.to_string_lossy().as_ref()
    );

    use sha2::{Digest, Sha256};
    let expected_hash = format!("{:x}", Sha256::digest(&jsonl));
    assert_eq!(report["batches_jsonl_sha256"], expected_hash);

    let second_run = Command::new(env!(
        "CARGO_BIN_EXE_billiards-perception-receive-detection-batches"
    ))
    .args([
        "--bind-address",
        "127.0.0.1:0",
        "--expected-batches",
        "1",
        "--maximum-message-bytes",
        "65536",
        "--accept-timeout-ms",
        "10",
        "--session-timeout-ms",
        "10",
        "--read-timeout-ms",
        "10",
        "--batches-jsonl",
    ])
    .arg(&jsonl_path)
    .output()
    .await
    .unwrap();
    assert!(!second_run.status.success());
}
