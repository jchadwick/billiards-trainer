use std::time::Duration;

use billiards_perception::performance_audit_receiver::{connect_and_collect, StateReceiverConfig};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn state_message(sequence: u64, frame_id: &str) -> String {
    state_message_with_status(sequence, frame_id, "fresh", "observed")
}

fn state_message_with_status(
    sequence: u64,
    frame_id: &str,
    freshness: &str,
    ball_positions_status: &str,
) -> String {
    json!({
        "type": "state",
        "sequence": sequence,
        "data": {
            "sequence": sequence,
            "balls": [{"id": "cue"}, {"id": "object"}],
            "observation_status": {
                "freshness_status": freshness,
                "ball_positions_status": ball_positions_status,
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
async fn readiness_receiver_stops_on_the_first_fresh_observed_localization() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        socket
            .send(Message::Text(
                state_message_with_status(1, "100", "stale", "observed").into(),
            ))
            .await
            .unwrap();
        socket
            .send(Message::Text(state_message(2, "101").into()))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
    });

    let config = StateReceiverConfig {
        duration: Duration::from_secs(2),
        connect_timeout: Duration::from_secs(1),
        application_heartbeat_interval: Duration::from_secs(20),
        summary_interval: Duration::from_secs(1),
        maximum_message_bytes: 4096,
        maximum_samples: 1,
        resource_sample_process_id: None,
        resource_sample_interval: Duration::from_secs(1),
        subscriber_delay: Duration::ZERO,
    };
    let output = billiards_perception::performance_audit_receiver::connect_and_wait_for_first_fresh_observed_localization(
        &format!("ws://{address}/state"),
        &config,
    )
    .await
    .unwrap();
    let close = server.await.unwrap();

    assert!(matches!(close, Message::Close(_)));
    assert_eq!(
        output.report.termination_reason,
        "first_fresh_observed_localization"
    );
    assert_eq!(output.report.counts.state_messages, 2);
    assert_eq!(output.retained_samples.len(), 1);
    assert_eq!(
        output.retained_samples[0].source_frame_id.as_deref(),
        Some("101")
    );
    assert!(output.report.elapsed_seconds < config.duration.as_secs_f64());
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
        application_heartbeat_interval: Duration::from_secs(20),
        summary_interval: Duration::from_secs(1),
        maximum_message_bytes: 4096,
        maximum_samples: 16,
        resource_sample_process_id: None,
        resource_sample_interval: Duration::from_secs(1),
        subscriber_delay: Duration::ZERO,
    };
    let output = connect_and_collect(&format!("ws://{address}/state"), &config)
        .await
        .unwrap();
    server.await.unwrap();

    assert_eq!(output.report.termination_reason, "peer_closed");
    assert_eq!(output.report.counts.state_messages, 2);
    assert_eq!(output.report.freshness.fresh_state_messages, 2);
    assert_eq!(output.report.freshness.fresh_unique_source_observations, 2);
    assert_eq!(output.report.freshness.unique_source_frame_updates, 2);
    assert_eq!(
        output.report.freshness.unique_observed_source_observations,
        2
    );
    assert_eq!(output.report.completeness.complete_state_messages, 2);
    assert_eq!(output.report.completeness.incomplete_state_messages, 0);
    assert_eq!(
        output.report.component_observation_status_counts["balls"]["observed"],
        2
    );
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
    assert!(output.trajectory.is_none());
}

#[tokio::test]
async fn negotiates_application_subscription_and_answers_json_heartbeats() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = accept_async(stream).await.unwrap();
        socket
            .send(Message::Text(
                json!({"type": "connection", "data": {"client_id": "audit-client"}})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();

        let subscribe = tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let Message::Text(subscribe) = subscribe else {
            panic!("expected a text subscription request");
        };
        let subscribe: serde_json::Value = serde_json::from_str(&subscribe).unwrap();
        assert_eq!(subscribe["type"], "subscribe");
        assert_eq!(subscribe["data"]["streams"], json!(["state"]));

        socket
            .send(Message::Text(
                json!({"type": "subscribed", "data": {"streams": ["state"]}})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        socket
            .send(Message::Text(state_message(7, "700").into()))
            .await
            .unwrap();
        socket
            .send(Message::Text(
                json!({"type": "ping", "data": {}}).to_string().into(),
            ))
            .await
            .unwrap();

        let mut application_ping_responses = 0;
        while let Some(message) = tokio::time::timeout(Duration::from_secs(1), socket.next())
            .await
            .unwrap()
        {
            let message = message.unwrap();
            if let Message::Text(text) = message {
                let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                if value["type"] == "ping" {
                    application_ping_responses += 1;
                }
            }
        }
        application_ping_responses
    });

    let config = StateReceiverConfig {
        duration: Duration::from_millis(150),
        connect_timeout: Duration::from_secs(1),
        application_heartbeat_interval: Duration::from_millis(25),
        summary_interval: Duration::from_secs(1),
        maximum_message_bytes: 4096,
        maximum_samples: 16,
        resource_sample_process_id: None,
        resource_sample_interval: Duration::from_secs(1),
        subscriber_delay: Duration::ZERO,
    };
    let output = connect_and_collect(&format!("ws://{address}/api/v1/websocket/ws"), &config)
        .await
        .unwrap();
    let application_ping_responses = server.await.unwrap();

    assert_eq!(output.report.counts.state_messages, 1);
    assert_eq!(
        output.retained_samples[0].source_frame_id.as_deref(),
        Some("700")
    );
    // Subscription acknowledgements are now consumed outside the measurement
    // window, while the server's application ping remains a measured message.
    assert!(output.report.counts.non_state_messages >= 1);
    assert!(application_ping_responses >= 1);
}
