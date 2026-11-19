use std::time::{Duration, Instant};

use billiards_perception::producer_diagnostics::{
    serve, DiagnosticsRegistry, ProducerReporter, ReporterLaunch,
};
use billiards_perception::producer_diagnostics_contract::{
    DiagnosticsConfig, ModelState, ProducerDiagnosticsReport, ProducerExitReason,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

fn fixture() -> ProducerDiagnosticsReport {
    ProducerDiagnosticsReport::decode(
        include_bytes!("../contracts/producer-diagnostics/v1/active.json"),
        65_536,
    )
    .unwrap()
}

#[test]
fn receiver_rejects_old_generations_sequences_and_process_switches_and_clears_active_model() {
    let registry = DiagnosticsRegistry::default();
    let mut report = fixture();
    assert!(registry.accept(report.clone(), Instant::now()).is_err());
    registry.begin_generation(report.generation_id.clone());
    let now = Instant::now();
    registry.accept(report.clone(), now).unwrap();
    assert!(registry.accept(report.clone(), now).is_err());
    assert!(registry
        .active_report_at(now, Duration::from_secs(3))
        .unwrap()
        .metadata
        .model
        .is_some());
    assert!(registry
        .active_report_at(now + Duration::from_secs(3), Duration::from_secs(3))
        .is_none());
    report.sequence += 1;
    report.process_id = "different-process".into();
    assert!(registry.accept(report.clone(), now).is_err());
    registry.begin_generation("replacement-generation".into());
    assert!(registry.snapshot().report.is_none());
    assert_eq!(
        registry
            .snapshot()
            .last_known
            .as_ref()
            .map(|last| last.report.generation_id.as_str()),
        Some(report.generation_id.as_str())
    );
    assert!(registry.accept(report, now).is_err());
    let mut replacement = fixture();
    replacement.generation_id = "replacement-generation".into();
    registry.accept(replacement.clone(), now).unwrap();
    registry.retire_generation("retired-old-process", ProducerExitReason::Unknown);
    assert!(registry.snapshot().report.is_some());
    registry.retire_generation(&replacement.generation_id, ProducerExitReason::GracefulStop);
    assert!(registry.snapshot().report.is_none());
    assert_eq!(
        registry.snapshot().exit_reason,
        Some(ProducerExitReason::GracefulStop)
    );
    replacement.sequence += 1;
    assert!(registry.accept(replacement, now).is_err());
}

async fn next_matching(
    updates: &mut watch::Receiver<billiards_perception::producer_diagnostics::ReceivedDiagnostics>,
    expected: ModelState,
) -> ProducerDiagnosticsReport {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let snapshot = updates.borrow_and_update().clone();
            if let Some(report) = snapshot.report {
                if report.metadata.model_state == expected {
                    return report;
                }
            }
            updates.changed().await.unwrap();
        }
    })
    .await
    .unwrap()
}

async fn next_failure_report(
    updates: &mut watch::Receiver<billiards_perception::producer_diagnostics::ReceivedDiagnostics>,
) -> ProducerDiagnosticsReport {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let snapshot = updates.borrow_and_update().clone();
            if let Some(report) = snapshot.report {
                if report.metadata.failure_code.is_some() {
                    return report;
                }
            }
            updates.changed().await.unwrap();
        }
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn reporter_streams_loading_active_failure_and_heartbeat_without_any_scene() {
    let policy = DiagnosticsConfig {
        heartbeat_interval_ms: 20,
        report_expiry_ms: 500,
        ..Default::default()
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    assert_ne!(address.port(), 0);
    let registry = DiagnosticsRegistry::default();
    registry.begin_generation("generation-test".into());
    let mut updates = registry.subscribe();
    let (stop, shutdown) = watch::channel(false);
    let listener_task = tokio::spawn(serve(
        listener,
        registry.clone(),
        policy.clone(),
        2,
        shutdown,
    ));
    let reporter = ProducerReporter::start(ReporterLaunch {
        address,
        generation_id: "generation-test".into(),
        policy,
        connect_timeout_ms: 100,
        write_timeout_ms: 100,
        reconnect_initial_delay_ms: 10,
        reconnect_maximum_delay_ms: 50,
    })
    .unwrap();
    let loading = next_matching(&mut updates, ModelState::Loading).await;
    assert!(loading.metadata.model.is_none());
    let handle = reporter.handle();
    handle.set_engine_identity("engine-test".into());
    handle.set_capture_session("capture-test".into());
    handle.activate_model(fixture().metadata.model.unwrap());
    let active = next_matching(&mut updates, ModelState::Active).await;
    assert!(active.sequence > loading.sequence);
    assert_eq!(
        active.metadata.engine_instance_id.as_deref(),
        Some("engine-test")
    );
    assert_eq!(active.metadata.build.package_name, env!("CARGO_PKG_NAME"));
    let gateway_build = billiards_perception::producer_diagnostics_contract::BuildInfo {
        git_revision: Some("a".repeat(40)),
        ..active.metadata.build.clone()
    };
    // The receiver preserves producer build information, never replacing it
    // with a potentially differently built gateway's version/revision.
    assert_ne!(gateway_build, active.metadata.build);
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            updates.changed().await.unwrap();
            if updates
                .borrow_and_update()
                .report
                .as_ref()
                .unwrap()
                .sequence
                > active.sequence
            {
                break;
            }
        }
    })
    .await
    .unwrap();
    handle.fail();
    let failed = next_failure_report(&mut updates).await;
    assert_eq!(failed.metadata.model_state, ModelState::Active);
    assert!(failed.metadata.model.is_some());
    assert_eq!(
        failed.metadata.failure_code.as_deref(),
        Some("producer_failed")
    );
    reporter.finish();
    stop.send(true).unwrap();
    listener_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn bad_length_and_unknown_generation_do_not_poison_listener() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let registry = DiagnosticsRegistry::default();
    let (stop, shutdown) = watch::channel(false);
    let task = tokio::spawn(serve(
        listener,
        registry.clone(),
        DiagnosticsConfig::default(),
        2,
        shutdown,
    ));
    for length in [0u32, 65_537, u32::MAX] {
        let mut socket = TcpStream::connect(address).await.unwrap();
        socket.write_u32(length).await.unwrap();
        let mut byte = [0];
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), socket.read(&mut byte))
                .await
                .unwrap()
                .unwrap(),
            0
        );
    }
    let report = fixture();
    let bytes = serde_json::to_vec(&report).unwrap();
    let mut socket = TcpStream::connect(address).await.unwrap();
    socket.write_u32(bytes.len() as u32).await.unwrap();
    socket.write_all(&bytes).await.unwrap();
    let mut byte = [0];
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), socket.read(&mut byte))
            .await
            .unwrap()
            .unwrap(),
        0
    );
    assert!(registry.snapshot().report.is_none());
    registry.begin_generation(report.generation_id.clone());
    // A new valid reporter remains usable after rejected clients/headers.
    let mut updates = registry.subscribe();
    let reporter = ProducerReporter::start(ReporterLaunch {
        address,
        generation_id: report.generation_id,
        policy: DiagnosticsConfig::default(),
        connect_timeout_ms: 100,
        write_timeout_ms: 100,
        reconnect_initial_delay_ms: 10,
        reconnect_maximum_delay_ms: 50,
    })
    .unwrap();
    next_matching(&mut updates, ModelState::Loading).await;
    reporter.finish();
    stop.send(true).unwrap();
    task.await.unwrap().unwrap();
}

#[test]
fn launch_requires_allocated_loopback_and_bounded_transport_policies() {
    let mut launch = ReporterLaunch {
        address: "127.0.0.1:1234".parse().unwrap(),
        generation_id: "generation".into(),
        policy: DiagnosticsConfig::default(),
        connect_timeout_ms: 10,
        write_timeout_ms: 10,
        reconnect_initial_delay_ms: 10,
        reconnect_maximum_delay_ms: 100,
    };
    launch.validate().unwrap();
    launch.address = "0.0.0.0:1234".parse().unwrap();
    assert!(launch.validate().is_err());
    launch.address = "127.0.0.1:0".parse().unwrap();
    assert!(launch.validate().is_err());
    launch.address = "127.0.0.1:1234".parse().unwrap();
    launch.write_timeout_ms = 0;
    assert!(launch.validate().is_err());
}

#[tokio::test]
async fn actual_producer_reports_startup_failure_before_configuration_or_scenes() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let registry = DiagnosticsRegistry::default();
    registry.begin_generation("startup-failure".into());
    let mut updates = registry.subscribe();
    let (stop, shutdown) = watch::channel(false);
    let listener_task = tokio::spawn(serve(
        listener,
        registry.clone(),
        DiagnosticsConfig::default(),
        2,
        shutdown,
    ));
    let launch = ReporterLaunch {
        address,
        generation_id: "startup-failure".into(),
        policy: DiagnosticsConfig::default(),
        connect_timeout_ms: 100,
        write_timeout_ms: 100,
        reconnect_initial_delay_ms: 10,
        reconnect_maximum_delay_ms: 50,
    };
    let result = tokio::process::Command::new(env!("CARGO_BIN_EXE_billiards-perception-live"))
        .arg("--not-a-valid-option")
        .env(
            billiards_perception::producer_diagnostics::REPORTER_ENV,
            serde_json::to_string(&launch).unwrap(),
        )
        .stdin(std::process::Stdio::null())
        .output()
        .await
        .unwrap();
    assert!(!result.status.success());
    let report = next_failure_report(&mut updates).await;
    assert_eq!(report.metadata.model_state, ModelState::Loading);
    assert!(report.metadata.model.is_none());
    assert!(report.metadata.capture_session_id.is_none());
    assert_eq!(
        report.metadata.failure_code.as_deref(),
        Some("producer_failed")
    );
    stop.send(true).unwrap();
    listener_task.await.unwrap().unwrap();
}
