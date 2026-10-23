use std::path::PathBuf;
use std::time::Duration;

use billiards_perception::performance_audit_receiver::StateReceiverConfig;
use billiards_perception::performance_audit_runner::{
    run_state_audit, ManagedServerConfig, StateAuditRunConfig,
};
use futures_util::SinkExt;
use serde_json::Value;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn state_message(sequence: u64, frame: u64) -> String {
    state_message_with_ball_status(sequence, frame, "observed")
}

fn state_message_with_ball_status(sequence: u64, frame: u64, ball_status: &str) -> String {
    serde_json::json!({
        "type": "state",
        "sequence": sequence,
        "data": {
            "sequence": sequence,
            "balls": [],
            "observation_status": {
                "engine_instance_id": "vision-runner-test",
                "source_session_id": "session-runner-test",
                "source_frame_id": format!("frame-{frame}"),
                "source_frame_number": frame,
                "freshness_status": "fresh",
                "ball_positions_status": ball_status,
                "component_observation_status": {"balls": ball_status}
            }
        }
    })
    .to_string()
}

fn run_config(url: String, output_root: std::path::PathBuf) -> StateAuditRunConfig {
    StateAuditRunConfig {
        url,
        warmup: Duration::ZERO,
        measured: Duration::from_secs(1),
        repetitions: 2,
        soak: false,
        soak_duration: Duration::from_secs(1),
        output_root,
        receiver: StateReceiverConfig {
            duration: Duration::from_secs(1),
            connect_timeout: Duration::from_secs(2),
            application_heartbeat_interval: Duration::from_secs(1),
            summary_interval: Duration::from_secs(1),
            maximum_message_bytes: 1_048_576,
            maximum_samples: 8,
            resource_sample_process_id: None,
            resource_sample_interval: Duration::from_secs(1),
        },
        managed_server: None,
        live_pipeline_timing_audit: None,
        sample_managed_live_process_resources: false,
    }
}

#[tokio::test]
async fn repeatable_runner_writes_bounded_reports_details_and_provenance() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let peer = tokio::spawn(async move {
        for frame in 1..=2 {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            socket
                .send(Message::Text(state_message(frame, frame).into()))
                .await
                .unwrap();
            socket.close(None).await.unwrap();
        }
    });
    let directory = tempdir().unwrap();
    let config = run_config(
        format!("ws://{address}/state"),
        directory.path().join("audits"),
    );

    let run_directory = run_state_audit(&config).await.unwrap();
    peer.await.unwrap();

    let summary: Value =
        serde_json::from_slice(&std::fs::read(run_directory.join("summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["schema_version"], 5);
    assert_eq!(summary["mode"], "repeated");
    assert_eq!(summary["measured_runs"].as_array().unwrap().len(), 2);
    assert!(summary["environment"]["working_tree_dirty"].is_boolean());
    assert_eq!(
        summary["input"]["content_hash_status"],
        "unavailable_for_network_stream"
    );
    assert!(summary["limitations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item.as_str().unwrap().contains("accuracy")));

    for repetition in 1..=2 {
        let report_path = run_directory.join(format!("runs/run-{repetition:03}.json"));
        let report: Value = serde_json::from_slice(&std::fs::read(report_path).unwrap()).unwrap();
        assert_eq!(report["report"]["counts"]["state_messages"], 1);
        assert_eq!(
            report["report"]["freshness"]["unique_observed_source_observations"],
            1
        );
        let details =
            std::fs::read_to_string(run_directory.join(format!("runs/run-{repetition:03}.jsonl")))
                .unwrap();
        let sample: Value = serde_json::from_str(details.trim()).unwrap();
        assert_eq!(sample["state_complete"], true);
        assert!(sample["source_frame_number"].is_number());
    }
    let markdown = std::fs::read_to_string(run_directory.join("report.md")).unwrap();
    assert!(markdown.contains("Observed source frames"));
    assert!(markdown.contains("accuracy"));
}

#[tokio::test]
async fn warmup_is_reported_as_excluded_and_soak_uses_one_configured_window() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let peer = tokio::spawn(async move {
        for frame in 1..=2 {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            socket
                .send(Message::Text(state_message(frame, frame).into()))
                .await
                .unwrap();
            socket.close(None).await.unwrap();
        }
    });
    let directory = tempdir().unwrap();
    let mut config = run_config(
        format!("ws://{address}/state"),
        directory.path().join("audits"),
    );
    config.warmup = Duration::from_secs(1);
    config.repetitions = 12;
    config.soak = true;
    config.soak_duration = Duration::from_secs(1);

    let run_directory = run_state_audit(&config).await.unwrap();
    peer.await.unwrap();

    let summary: Value =
        serde_json::from_slice(&std::fs::read(run_directory.join("summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["mode"], "soak");
    assert_eq!(summary["configuration"]["repetitions"], 1);
    assert_eq!(summary["configuration"]["measurement_window_seconds"], 1.0);
    assert!(summary["warmup"]["measurements_excluded"]
        .as_bool()
        .unwrap());
    assert_eq!(summary["warmup"]["state_messages"], 1);
    assert_eq!(summary["measured_runs"].as_array().unwrap().len(), 1);
    assert!(!run_directory.join("runs/run-002.json").exists());
}

#[tokio::test]
async fn invalid_configuration_does_not_create_the_output_root() {
    let directory = tempdir().unwrap();
    let output_root = directory.path().join("must-not-exist");
    let mut config = run_config("ws://127.0.0.1:8080/state".to_owned(), output_root.clone());
    config.repetitions = 0;

    assert!(run_state_audit(&config).await.is_err());
    assert!(!output_root.exists());
}

#[tokio::test]
async fn connection_failure_writes_an_explicit_failure_artifact() {
    let directory = tempdir().unwrap();
    let config = run_config(
        "ws://127.0.0.1:1/state".to_owned(),
        directory.path().join("audits"),
    );

    assert!(run_state_audit(&config).await.is_err());
    let run_directory = std::fs::read_dir(config.output_root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let failure: Value =
        serde_json::from_slice(&std::fs::read(run_directory.join("failure.json")).unwrap())
            .unwrap();
    assert_eq!(failure["tool"], "billiards-performance-audit-run");
    assert!(failure["error"]
        .as_str()
        .unwrap()
        .contains("measured run 1"));
}

#[cfg(unix)]
#[tokio::test]
async fn managed_server_waits_for_fresh_observed_state_then_reaps_child() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let peer = tokio::spawn(async move {
        // The first connected state is failed and must not open the gate. The
        // second is the first fresh observed state; the remaining two belong
        // to the measured repetitions.
        for frame in 1..=4 {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let ball_status = if frame == 1 { "failed" } else { "observed" };
            socket
                .send(Message::Text(
                    state_message_with_ball_status(frame, frame, ball_status).into(),
                ))
                .await
                .unwrap();
            socket.close(None).await.unwrap();
        }
    });
    let directory = tempdir().unwrap();
    let mut config = run_config(
        format!("ws://{address}/api/v1/websocket/ws"),
        directory.path().join("audits"),
    );
    config.repetitions = 2;
    config.managed_server = Some(ManagedServerConfig {
        executable: PathBuf::from("sleep"),
        arguments: vec!["60".to_owned()],
        working_directory: None,
        startup_timeout: Duration::from_secs(2),
        readiness_probe_interval: Duration::from_millis(20),
        shutdown_timeout: Duration::from_secs(1),
    });

    let run_directory = run_state_audit(&config).await.unwrap();
    peer.await.unwrap();

    let summary: Value =
        serde_json::from_slice(&std::fs::read(run_directory.join("summary.json")).unwrap())
            .unwrap();
    let server = &summary["managed_server"];
    assert_eq!(server["executable"], "sleep");
    assert!(server["process_id"].as_u64().is_some());
    assert!(server["readiness_probe_attempts"].as_u64().unwrap() >= 2);
    assert_eq!(server["first_observed_source_frame_id"], "frame-2");
    assert_eq!(server["shutdown"]["status"], "terminated_gracefully");
    assert_eq!(server["shutdown"]["signal"], libc::SIGTERM);
    assert_eq!(server["shutdown"]["forced"], false);
}

#[cfg(unix)]
#[tokio::test]
async fn managed_server_startup_timeout_writes_failure_and_stops_child() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let directory = tempdir().unwrap();
    let mut config = run_config(
        format!("ws://{address}/api/v1/websocket/ws"),
        directory.path().join("audits"),
    );
    config.repetitions = 1;
    config.managed_server = Some(ManagedServerConfig {
        executable: PathBuf::from("sleep"),
        arguments: vec!["60".to_owned()],
        working_directory: None,
        startup_timeout: Duration::from_millis(80),
        readiness_probe_interval: Duration::from_millis(20),
        shutdown_timeout: Duration::from_secs(1),
    });

    assert!(run_state_audit(&config).await.is_err());
    let run_directory = std::fs::read_dir(config.output_root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let failure: Value =
        serde_json::from_slice(&std::fs::read(run_directory.join("failure.json")).unwrap())
            .unwrap();
    let server = &failure["managed_server"];
    let process_id = server["process_id"].as_i64().unwrap() as libc::pid_t;
    assert_eq!(server["shutdown"]["status"], "terminated_gracefully");
    assert_eq!(server["shutdown"]["signal"], libc::SIGTERM);
    // The child has been waited and reaped, so the PID must no longer exist.
    // SAFETY: signal zero performs a process-existence check without signaling.
    assert_eq!(unsafe { libc::kill(process_id, 0) }, -1);
}
