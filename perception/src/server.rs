//! Rust HTTP/WebSocket gateway for live scene snapshots.
//!
//! Scene ingress is accepted only when the gateway can resolve the active
//! Rust-owned capture settings and validate the producer's request-bound
//! readiness acknowledgement. The gateway can supervise the Rust detector
//! directly; the browser remains HTML/JavaScript/CSS and keeps its ordinary
//! JSON-over-HTTP/WebSocket contract.

use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Extension, Path, Query, State};
use axum::http::header::CACHE_CONTROL;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use base64::Engine as _;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex as AsyncMutex, RwLock as AsyncRwLock, Semaphore};
use tokio::time::{sleep, timeout};

use crate::browser_ball_placement::{
    BallSearchTarget, PlacementObservation, PlacementRequest, BALL_PLACEMENT_REQUEST_V1,
    MAXIMUM_PLACEMENT_MESSAGE_BYTES,
};
use crate::browser_capture_archive::CaptureMode;
use crate::browser_diagnostics_contract::{
    BrowserDiagnosticsSnapshot, BrowserDiagnosticsStage, BrowserDiagnosticsStreamPolicy,
    DiagnosticsStageName, GatewayRuntimeSnapshot, ProducerLastKnownSnapshot,
    ProducerReportSnapshot, ProducerReportState, StageEvidenceSource,
    BROWSER_DIAGNOSTICS_SCHEMA_VERSION,
};
use crate::capture_settings_wire::{parse_capture_settings_json, ResolvedCaptureSettings};
use crate::configuration_authority::{AuthorityLimits, ConfigurationAuthority};
use crate::frame_preview::{
    FramePreviewDescriptorV1, FramePreviewResponseHeaderV1, FramePreviewStatusV1,
    PREVIEW_REQUEST_V1,
};
use crate::framing::{read_framed_message, write_framed_message};
use crate::game_session::GameSessionStore;
use crate::game_session_http::GameSessionHttpState;
use crate::live_audit::{
    LiveAuditConfig, LiveAuditExporter, LiveAuditGatewayWebSocketSend, LiveAuditRecorder,
};
use crate::live_handshake::{validate_live_v3_hello, ValidatedLiveHandshake};
use crate::live_supervisor::LiveProcessConfig;
use crate::producer_diagnostics::DiagnosticsRegistry;
use crate::producer_diagnostics_contract::{
    BuildInfo, DiagnosticsConfig, ProducerStageName, StageStatus,
};
use crate::public_state::PublicStateConfig;
use crate::public_state_lifecycle::{
    CurrentSceneState, IngestOutcome, PerceptionStateV1, SceneStateLifecycle, SourceStateMetadata,
    StateFreshness, StateLifecycleConfig,
};
use crate::SceneSnapshotV1;

const LIVE_PROTOCOL_VERSION: u8 = 3;
const MAX_IDENTIFIER_CHARACTERS: usize = 128;
const DEFAULT_HTTP_PORT: u16 = 8001;
const DEFAULT_SCENE_PORT: u16 = 49_321;
const DEFAULT_MAX_SCENE_BYTES: usize = 262_144;
const MAXIMUM_SCENE_BYTES_LIMIT: usize = 32 * 1024 * 1024;
const DEFAULT_READ_TIMEOUT_MS: u64 = 2_000;
const DEFAULT_STALE_AFTER_MS: u64 = 100;
const DEFAULT_EXPIRE_AFTER_MS: u64 = 1_000;
const DEFAULT_WS_SEND_DEADLINE_MS: u64 = 250;
const DEFAULT_MAX_WEBSOCKET_CLIENTS: usize = 64;
const POSITION_RETENTION_CONTEXT_FIELDS: [&str; 6] = [
    "engine_instance_id",
    "source_session_id",
    "configuration_revision",
    "rectification_revision",
    "calibration_revision",
    "calibration_state",
];
const DIAGNOSTICS_WEBSOCKET_PATH: &str = "/api/v1/diagnostics/ws";
pub const DEFAULT_MAX_CONFIGURATION_REQUESTS: usize = 4;

/// Runtime options for the independently launchable gateway.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub http_bind: IpAddr,
    pub http_port: u16,
    pub scene_bind: IpAddr,
    pub scene_port: u16,
    /// Deployment-owned root that confines configuration-selected data files.
    /// `None` uses the service process working directory.
    pub project_root: Option<PathBuf>,
    pub maximum_scene_bytes: usize,
    /// Active unified config required for live-v3 ingress and configuration routes.
    pub configuration_path: Option<PathBuf>,
    pub tcp_read_timeout: Duration,
    pub stale_after: Duration,
    pub expire_after: Duration,
    pub websocket_send_deadline: Duration,
    pub maximum_websocket_clients: usize,
    /// Maximum concurrent operations that may touch the configuration authority.
    pub maximum_configuration_requests: usize,
    /// Optional prebuilt HTML/JS/CSS asset directory served by this Rust process.
    pub web_root: Option<PathBuf>,
}

/// Optional live detector launch policy. Tests and offline replay tools can
/// exercise their own boundaries without launching camera hardware.
#[derive(Debug, Clone, Default)]
pub struct LiveProcessLaunchOptions {
    pub enabled: bool,
    pub executable_path: Option<PathBuf>,
    pub capture_helper_path: Option<PathBuf>,
    /// Optional bounded timing side channel. `None` keeps the detector's
    /// recorder, queue, and exporter completely disabled.
    pub audit_config: Option<LiveAuditConfig>,
    /// Independent gateway WebSocket timing channel. This can be enabled even
    /// when the detector itself is externally managed or has no timing audit.
    pub gateway_audit_config: Option<GatewayAuditConfig>,
}

/// Configuration for the separately retained gateway WebSocket audit stream.
#[derive(Debug, Clone)]
pub struct GatewayAuditConfig {
    pub exporter: LiveAuditConfig,
    pub run_id: String,
}

impl GatewayAuditConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.exporter
            .validate()
            .map_err(|error| format!("invalid gateway audit exporter configuration: {error}"))?;
        if self.run_id.trim().is_empty()
            || self.run_id.len() > 128
            || self.run_id.chars().any(char::is_control)
        {
            return Err("gateway audit run ID must contain 1 to 128 non-control characters".into());
        }
        Ok(())
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            http_bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
            http_port: DEFAULT_HTTP_PORT,
            scene_bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
            scene_port: DEFAULT_SCENE_PORT,
            project_root: None,
            maximum_scene_bytes: DEFAULT_MAX_SCENE_BYTES,
            configuration_path: None,
            tcp_read_timeout: Duration::from_millis(DEFAULT_READ_TIMEOUT_MS),
            stale_after: Duration::from_millis(DEFAULT_STALE_AFTER_MS),
            expire_after: Duration::from_millis(DEFAULT_EXPIRE_AFTER_MS),
            websocket_send_deadline: Duration::from_millis(DEFAULT_WS_SEND_DEADLINE_MS),
            maximum_websocket_clients: DEFAULT_MAX_WEBSOCKET_CLIENTS,
            maximum_configuration_requests: DEFAULT_MAX_CONFIGURATION_REQUESTS,
            web_root: None,
        }
    }
}

impl ServerConfig {
    /// Reject unsafe or internally inconsistent options before binding sockets.
    pub fn validate(&self) -> Result<(), String> {
        if !self.scene_bind.is_loopback() {
            return Err("scene bind address must be loopback".into());
        }
        if !(1..=MAXIMUM_SCENE_BYTES_LIMIT).contains(&self.maximum_scene_bytes) {
            return Err(format!(
                "maximum scene bytes must be between 1 and {MAXIMUM_SCENE_BYTES_LIMIT}"
            ));
        }
        if self
            .configuration_path
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err("configuration path must name a file".into());
        }
        if self
            .project_root
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
        {
            return Err("project root must be an absolute directory path".into());
        }
        if self
            .web_root
            .as_ref()
            .is_some_and(|path| !path.is_absolute() || !path.is_dir())
        {
            return Err("web root must be an absolute existing directory".into());
        }
        validate_duration(
            "TCP read timeout",
            self.tcp_read_timeout,
            Duration::from_millis(10),
            Duration::from_secs(300),
        )?;
        validate_duration(
            "stale-after policy",
            self.stale_after,
            Duration::from_millis(10),
            Duration::from_secs(600),
        )?;
        validate_duration(
            "expiry policy",
            self.expire_after,
            Duration::from_millis(20),
            Duration::from_secs(86_400),
        )?;
        if self.expire_after <= self.stale_after {
            return Err("expiry policy must be greater than stale-after policy".into());
        }
        validate_duration(
            "WebSocket send deadline",
            self.websocket_send_deadline,
            Duration::from_millis(1),
            Duration::from_secs(60),
        )?;
        if !(1..=4096).contains(&self.maximum_websocket_clients) {
            return Err("maximum WebSocket clients must be between 1 and 4096".into());
        }
        if !(1..=256).contains(&self.maximum_configuration_requests) {
            return Err("maximum configuration requests must be between 1 and 256".into());
        }
        Ok(())
    }

    fn lifecycle_config(&self) -> StateLifecycleConfig {
        StateLifecycleConfig {
            stale_after_ns: self.stale_after.as_nanos(),
            expire_after_ns: self.expire_after.as_nanos(),
            ..StateLifecycleConfig::default()
        }
    }
}

fn validate_duration(
    name: &str,
    value: Duration,
    minimum: Duration,
    maximum: Duration,
) -> Result<(), String> {
    if value < minimum || value > maximum {
        return Err(format!(
            "{name} must be between {} ms and {} ms",
            minimum.as_millis(),
            maximum.as_millis()
        ));
    }
    Ok(())
}

#[derive(Clone)]
pub struct Gateway {
    inner: Arc<GatewayInner>,
}

struct GatewayInner {
    config: ServerConfig,
    configuration_authority: Option<ConfigurationAuthority>,
    game_session_store: Option<GameSessionStore>,
    started_at: Instant,
    state: Mutex<IngressState>,
    latest: watch::Sender<Option<Arc<PublishedState>>>,
    shutdown: watch::Sender<bool>,
    producer_connected: AtomicBool,
    producer_connection_updates: watch::Sender<bool>,
    producer_control_stream: Mutex<Option<std::net::TcpStream>>,
    websocket_slots: Arc<Semaphore>,
    configuration_slots: Arc<Semaphore>,
    frame_preview_stream: Option<FramePreviewStreamSettings>,
    frame_preview_requests: AsyncRwLock<()>,
    ball_placement_last_request: AsyncMutex<Option<Instant>>,
    configuration_mutation_lock: Arc<AsyncMutex<()>>,
    setup_reload: watch::Sender<()>,
    calibration_stills: Mutex<std::collections::VecDeque<(String, [u8; 32])>>,
    setup_enabled: AtomicBool,
    setup_error: Mutex<Option<String>>,
    gateway_audit: Option<Arc<GatewayAudit>>,
    diagnostics: DiagnosticsRegistry,
    diagnostics_policy: DiagnosticsConfig,
    gateway_instance_id: String,
    diagnostics_revision: AtomicU64,
    diagnostics_snapshots: watch::Sender<BrowserDiagnosticsSnapshot>,
    diagnostics_websocket_slots: Arc<Semaphore>,
}

#[derive(Debug, Clone)]
struct FramePreviewStreamSettings {
    address: SocketAddr,
    request_timeout: Duration,
    maximum_header_bytes: usize,
    maximum_jpeg_bytes: usize,
    calibration: Option<CalibrationPreviewLimits>,
}

#[derive(Debug, Clone)]
struct CalibrationPreviewLimits {
    maximum_request_bytes: usize,
    maximum_response_bytes: usize,
}

#[derive(Debug)]
struct IdentifiedMjpegFrame {
    descriptor: FramePreviewDescriptorV1,
    jpeg: Vec<u8>,
}

struct GatewayAudit {
    recorder: LiveAuditRecorder,
    run_id: String,
    gateway_instance_id: String,
    monotonic_clock_domain_id: String,
    next_client_id: AtomicU64,
}

struct GatewayWebSocketAuditClient {
    audit: Arc<GatewayAudit>,
    client_id: u64,
    previous_publication_sequence: Option<u64>,
    previous_source_identity: Option<(String, String, u64)>,
}

#[derive(Debug, Default, Deserialize)]
struct WebSocketQuery {
    retain_last_known_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
struct PositionRetentionContext {
    engine_instance_id: Option<Value>,
    source_session_id: Option<Value>,
    configuration_revision: Option<Value>,
    rectification_revision: Option<Value>,
    calibration_revision: Option<Value>,
    calibration_state: Option<Value>,
}

#[derive(Debug)]
struct RetainedBallLayout {
    balls: Vec<crate::public_state::PublicBall>,
    captured_at: Instant,
    context: PositionRetentionContext,
    original_status: Value,
    source_identity: Value,
}

/// Optional per-WebSocket presentation cache. It never mutates gateway state,
/// and it stores only the latest accepted observed layout for this client.
#[derive(Debug)]
struct WebSocketPositionRetention {
    duration: Duration,
    maximum_balls: usize,
    layout: Option<RetainedBallLayout>,
    unavailable_since: Option<Instant>,
}

impl WebSocketPositionRetention {
    fn new(duration: Duration, maximum_balls: usize) -> Self {
        Self {
            duration,
            maximum_balls,
            layout: None,
            unavailable_since: None,
        }
    }

    fn expiration_deadline(&self) -> Option<Instant> {
        self.layout
            .as_ref()
            .and(self.unavailable_since)
            .and_then(|unavailable_since| unavailable_since.checked_add(self.duration))
    }
}

struct IngressState {
    lifecycle: SceneStateLifecycle,
    producer_identity: Option<(String, String)>,
    last_transport_sequence: Option<u64>,
    /// V2 phase events have their own sequence space, separate from raw scenes.
    last_delivery_sequence: Option<u64>,
    publication_sequence: u64,
    published_signature: Option<(u64, u128, StateFreshness)>,
    localization_received_at: Option<DateTime<Utc>>,
    updated_at: Option<DateTime<Utc>>,
    live_source_geometry: Option<(u32, u32)>,
    live_configuration_revision: Option<u64>,
}

#[derive(Debug, Clone)]
struct PublishedState {
    current: Arc<CurrentSceneState>,
    published_at: DateTime<Utc>,
    snapshot_at_wall: DateTime<Utc>,
    snapshot_at_monotonic_ns: u128,
    published_at_monotonic_ns: u128,
    publication_sequence: u64,
}

impl Gateway {
    pub fn new(config: ServerConfig) -> Result<Self, String> {
        Self::new_with_audit(config, None)
    }

    fn new_with_audit(
        config: ServerConfig,
        audit: Option<(LiveAuditRecorder, String)>,
    ) -> Result<Self, String> {
        Self::new_with_audit_and_diagnostics(config, audit, DiagnosticsConfig::default())
    }

    fn new_with_audit_and_diagnostics(
        config: ServerConfig,
        audit: Option<(LiveAuditRecorder, String)>,
        diagnostics_policy: DiagnosticsConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        diagnostics_policy.validate()?;
        let configuration_authority = config
            .configuration_path
            .as_ref()
            .map(|path| ConfigurationAuthority::new(path, AuthorityLimits::default()))
            .transpose()?;
        let frame_preview_stream =
            match configured_frame_preview_stream(configuration_authority.as_ref()) {
                Ok(settings) => settings,
                Err(error) => {
                    eprintln!("identified_frame_stream_unavailable={error}");
                    None
                }
            };
        let game_session_store = configuration_authority.as_ref().and_then(|authority| {
            let root = config
                .project_root
                .clone()
                .or_else(|| std::env::current_dir().ok());
            match root {
                Some(root) => configured_game_session_store(authority, &root),
                None => {
                    eprintln!("game_session_store_unavailable=project_root_unavailable");
                    None
                }
            }
        });
        let lifecycle =
            SceneStateLifecycle::new(config.lifecycle_config(), PublicStateConfig::default())
                .map_err(|error| format!("invalid scene state lifecycle: {error}"))?;
        let (latest, _) = watch::channel(None);
        let (shutdown, _) = watch::channel(false);
        let gateway_instance_id = uuid::Uuid::new_v4().to_string();
        let (producer_connection_updates, _) = watch::channel(false);
        let initial_diagnostics =
            initial_browser_diagnostics_snapshot(&gateway_instance_id, &diagnostics_policy);
        let (diagnostics_snapshots, _) = watch::channel(initial_diagnostics);
        let diagnostics_websocket_slots =
            Arc::new(Semaphore::new(config.maximum_websocket_clients));
        let gateway_audit = audit.map(|(recorder, run_id)| {
            Arc::new(GatewayAudit {
                recorder,
                run_id,
                gateway_instance_id: gateway_instance_id.clone(),
                monotonic_clock_domain_id: uuid::Uuid::new_v4().to_string(),
                next_client_id: AtomicU64::new(1),
            })
        });
        Ok(Self {
            inner: Arc::new(GatewayInner {
                websocket_slots: Arc::new(Semaphore::new(config.maximum_websocket_clients)),
                configuration_slots: Arc::new(Semaphore::new(
                    config.maximum_configuration_requests,
                )),
                frame_preview_stream,
                frame_preview_requests: AsyncRwLock::new(()),
                ball_placement_last_request: AsyncMutex::new(None),
                configuration_mutation_lock: Arc::new(AsyncMutex::new(())),
                setup_reload: watch::channel(()).0,
                calibration_stills: Mutex::new(std::collections::VecDeque::new()),
                setup_enabled: AtomicBool::new(false),
                setup_error: Mutex::new(None),
                configuration_authority,
                game_session_store,
                config,
                started_at: Instant::now(),
                state: Mutex::new(IngressState {
                    lifecycle,
                    producer_identity: None,
                    last_transport_sequence: None,
                    last_delivery_sequence: None,
                    publication_sequence: 0,
                    published_signature: None,
                    localization_received_at: None,
                    updated_at: None,
                    live_source_geometry: None,
                    live_configuration_revision: None,
                }),
                latest,
                shutdown,
                producer_connected: AtomicBool::new(false),
                producer_connection_updates,
                producer_control_stream: Mutex::new(None),
                gateway_audit,
                diagnostics: DiagnosticsRegistry::default(),
                diagnostics_policy,
                gateway_instance_id,
                diagnostics_revision: AtomicU64::new(1),
                diagnostics_snapshots,
                diagnostics_websocket_slots,
            }),
        })
    }

    /// Producer metadata is independent of localization readiness and scene epochs.
    pub fn diagnostics_registry(&self) -> DiagnosticsRegistry {
        self.inner.diagnostics.clone()
    }

    fn publish_diagnostics_snapshot(&self) {
        let revision = self
            .inner
            .diagnostics_revision
            .fetch_add(1, Ordering::AcqRel)
            .saturating_add(1);
        let snapshot = self.assess_diagnostics_snapshot(revision);
        self.inner.diagnostics_snapshots.send_replace(snapshot);
    }

    fn assess_diagnostics_snapshot(&self, revision: u64) -> BrowserDiagnosticsSnapshot {
        let now = Instant::now();
        let producer = self.inner.diagnostics.snapshot();
        let report_age = producer
            .received_at
            .and_then(|received_at| now.checked_duration_since(received_at));
        let expiry = Duration::from_millis(self.inner.diagnostics_policy.report_expiry_ms);
        let report_fresh = !producer.exited
            && producer.report.is_some()
            && report_age.is_some_and(|age| age < expiry);
        let producer_state = if producer.exited {
            ProducerReportState::Stopped
        } else if report_fresh {
            ProducerReportState::Fresh
        } else if producer.report.is_some() {
            ProducerReportState::Expired
        } else {
            ProducerReportState::Unknown
        };
        let producer_snapshot = ProducerReportSnapshot {
            state: producer_state,
            generation_id: producer.generation_id.clone(),
            report_sequence: report_fresh
                .then(|| producer.report.as_ref().map(|report| report.sequence))
                .flatten(),
            report_age_ms: (report_fresh || producer.report.is_some())
                .then(|| report_age.map(duration_millis))
                .flatten(),
            expires_in_ms: report_fresh
                .then(|| expiry.saturating_sub(report_age.unwrap_or_default()))
                .map(duration_millis),
            metadata: report_fresh
                .then(|| {
                    producer
                        .report
                        .as_ref()
                        .map(|report| report.metadata.clone())
                })
                .flatten(),
            exit_reason: producer.exited.then_some(producer.exit_reason).flatten(),
            last_known: (!report_fresh)
                .then(|| {
                    producer
                        .last_known
                        .as_ref()
                        .map(|last_known| ProducerLastKnownSnapshot {
                            report: last_known.report.clone(),
                            report_age_ms: duration_millis(
                                now.saturating_duration_since(last_known.received_at),
                            ),
                        })
                })
                .flatten(),
        };
        let report = report_fresh.then_some(producer.report.as_ref()).flatten();
        let report_age_ms = report_age.map(duration_millis).unwrap_or_default();
        let mut stages = Vec::with_capacity(7);
        stages.push(stage_assessment(
            DiagnosticsStageName::GatewayService,
            StageStatus::Healthy,
            "gateway_running",
            StageEvidenceSource::GatewayRuntime,
            None,
            Some(0),
            None,
        ));
        for (producer_stage, browser_stage) in [
            (
                ProducerStageName::SourceLifecycle,
                DiagnosticsStageName::SourceLifecycle,
            ),
            (
                ProducerStageName::CaptureProgress,
                DiagnosticsStageName::CaptureProgress,
            ),
            (
                ProducerStageName::ModelInitialization,
                DiagnosticsStageName::ModelInitialization,
            ),
            (
                ProducerStageName::ScheduledPerceptionWork,
                DiagnosticsStageName::ScheduledPerceptionWork,
            ),
        ] {
            stages.push(self.assess_producer_stage(
                &producer,
                report,
                producer_stage,
                browser_stage,
                report_age_ms,
            ));
        }

        // Diagnostics read the latest accepted lifecycle snapshot without
        // triggering scene-watch publication. Existing scene freshness
        // behavior remains owned by its independent scheduler.
        let state = self.inner.latest.borrow().clone();
        let now_ns = self.monotonic_ns();
        stages.push(self.assess_scene_delivery(state.as_deref(), now_ns));
        stages.push(self.assess_localization(state.as_deref(), now_ns));
        let snapshot = BrowserDiagnosticsSnapshot {
            schema_version: BROWSER_DIAGNOSTICS_SCHEMA_VERSION,
            gateway: GatewayRuntimeSnapshot {
                instance_id: self.inner.gateway_instance_id.clone(),
                revision,
                assessed_at_utc: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
                uptime_ms: duration_millis(self.inner.started_at.elapsed()),
                build: BuildInfo::current(),
            },
            producer: producer_snapshot,
            stream: BrowserDiagnosticsStreamPolicy {
                heartbeat_interval_ms: self.inner.diagnostics_policy.stream_heartbeat_interval_ms,
                stale_after_ms: self.inner.diagnostics_policy.stream_stale_after_ms,
            },
            stages,
        };
        debug_assert!(snapshot.validate().is_ok());
        snapshot
    }

    fn assess_producer_stage(
        &self,
        producer: &crate::producer_diagnostics::ReceivedDiagnostics,
        report: Option<&crate::producer_diagnostics_contract::ProducerDiagnosticsReport>,
        producer_stage: ProducerStageName,
        browser_stage: DiagnosticsStageName,
        report_age_ms: u64,
    ) -> BrowserDiagnosticsStage {
        if let Some(observation) = report.and_then(|report| {
            report
                .stages
                .iter()
                .find(|stage| stage.stage == producer_stage)
        }) {
            let age = observation
                .last_progress_age_ms
                .map(|age| age.saturating_add(report_age_ms));
            let overdue = producer_stage == ProducerStageName::ScheduledPerceptionWork
                && observation.in_flight
                && observation.status == StageStatus::Healthy
                && observation.deadline_ms.is_some_and(|deadline| {
                    observation
                        .work_started_age_ms
                        .is_some_and(|age| age.saturating_add(report_age_ms) >= deadline)
                });
            let mut assessment = stage_assessment(
                browser_stage,
                if overdue {
                    StageStatus::Stalled
                } else {
                    observation.status
                },
                if overdue {
                    "active_work_deadline_exceeded"
                } else {
                    observation.reason_code.as_str()
                },
                StageEvidenceSource::ProducerReport,
                observation.detail.as_deref(),
                age,
                observation.deadline_ms,
            );
            assessment.in_flight = observation.in_flight;
            assessment.work_started_age_ms = observation
                .work_started_age_ms
                .map(|age| age.saturating_add(report_age_ms));
            assessment.expected_jobs = observation.expected_jobs;
            assessment.completed_jobs = observation.completed_jobs;
            assessment.failed_jobs = observation.failed_jobs;
            assessment.scheduler_state = observation.scheduler_state;
            assessment.active_job_kind = observation.active_job_kind;
            assessment.last_completion_age_ms = observation
                .last_completion_age_ms
                .map(|age| age.saturating_add(report_age_ms));
            assessment.last_failure_code = observation.last_failure_code.clone();
            assessment.last_failure_age_ms = observation
                .last_failure_age_ms
                .map(|age| age.saturating_add(report_age_ms));
            if !observation.in_flight {
                assessment.deadline_ms = None;
            }
            return assessment;
        }
        if producer.exited && producer_stage == ProducerStageName::SourceLifecycle {
            let (status, reason) = match producer.exit_reason.unwrap_or(
                crate::producer_diagnostics_contract::ProducerExitReason::Unknown,
            ) {
                crate::producer_diagnostics_contract::ProducerExitReason::GracefulStop => {
                    (StageStatus::Stopped, "producer_graceful_stop")
                }
                crate::producer_diagnostics_contract::ProducerExitReason::ProcessExitedSuccessfully => {
                    (StageStatus::Stopped, "producer_process_exited_successfully")
                }
                crate::producer_diagnostics_contract::ProducerExitReason::ProcessExitedNonzero => {
                    (StageStatus::Failed, "producer_process_exited_nonzero")
                }
                crate::producer_diagnostics_contract::ProducerExitReason::ProcessWaitFailed => {
                    (StageStatus::Failed, "producer_process_wait_failed")
                }
                crate::producer_diagnostics_contract::ProducerExitReason::SpawnFailed => {
                    (StageStatus::Failed, "producer_spawn_failed")
                }
                crate::producer_diagnostics_contract::ProducerExitReason::LaunchRequestFailed => {
                    (StageStatus::Failed, "producer_launch_request_failed")
                }
                crate::producer_diagnostics_contract::ProducerExitReason::ForcedTermination => {
                    (StageStatus::Failed, "producer_forced_termination")
                }
                crate::producer_diagnostics_contract::ProducerExitReason::Unknown => {
                    (StageStatus::Unknown, "producer_exit_outcome_unknown")
                }
            };
            return stage_assessment(
                browser_stage,
                status,
                reason,
                StageEvidenceSource::SupervisorRegistry,
                None,
                None,
                None,
            );
        }
        stage_assessment(
            browser_stage,
            StageStatus::Unknown,
            if producer.exited {
                "producer_process_exited"
            } else if producer.report.is_some() {
                "producer_report_expired"
            } else if producer.generation_id.is_some() {
                "producer_report_unavailable"
            } else {
                "producer_evidence_absent"
            },
            if producer.exited {
                StageEvidenceSource::SupervisorRegistry
            } else {
                StageEvidenceSource::ProducerReport
            },
            None,
            None,
            None,
        )
    }

    fn assess_scene_delivery(
        &self,
        state: Option<&PublishedState>,
        now_ns: u128,
    ) -> BrowserDiagnosticsStage {
        let Some(state) = state else {
            return stage_assessment(
                DiagnosticsStageName::SceneDelivery,
                StageStatus::Starting,
                "scene_not_received",
                StageEvidenceSource::SceneLifecycle,
                None,
                None,
                None,
            );
        };
        let age_ns = now_ns.saturating_sub(state.current.updated_monotonic_ns);
        let age_ms = u64::try_from(age_ns / 1_000_000).unwrap_or(u64::MAX);
        let age = Duration::from_nanos(u64::try_from(age_ns).unwrap_or(u64::MAX));
        let (status, reason) = if age < self.inner.config.stale_after {
            (StageStatus::Healthy, "scene_delivery_recent")
        } else if age < self.inner.config.expire_after {
            (StageStatus::Degraded, "scene_delivery_stale")
        } else {
            (StageStatus::Stalled, "scene_delivery_expired")
        };
        stage_assessment(
            DiagnosticsStageName::SceneDelivery,
            status,
            reason,
            StageEvidenceSource::SceneLifecycle,
            None,
            Some(age_ms),
            None,
        )
    }

    fn assess_localization(
        &self,
        state: Option<&PublishedState>,
        now_ns: u128,
    ) -> BrowserDiagnosticsStage {
        let Some(received_ns) =
            state.and_then(|state| state.current.localization_received_monotonic_ns)
        else {
            return stage_assessment(
                DiagnosticsStageName::Localization,
                StageStatus::Unknown,
                "localization_not_received",
                StageEvidenceSource::SceneLifecycle,
                None,
                None,
                None,
            );
        };
        let age_ns = now_ns.saturating_sub(received_ns);
        let age_ms = u64::try_from(age_ns / 1_000_000).unwrap_or(u64::MAX);
        let age = Duration::from_nanos(u64::try_from(age_ns).unwrap_or(u64::MAX));
        let (status, reason) = if age < self.inner.config.stale_after {
            (StageStatus::Healthy, "localization_fresh")
        } else if age < self.inner.config.expire_after {
            (StageStatus::Degraded, "localization_stale")
        } else {
            (StageStatus::Stalled, "localization_expired")
        };
        stage_assessment(
            DiagnosticsStageName::Localization,
            status,
            reason,
            StageEvidenceSource::SceneLifecycle,
            None,
            Some(age_ms),
            None,
        )
    }

    fn diagnostics_next_delay(&self) -> Duration {
        let now = Instant::now();
        let heartbeat =
            Duration::from_millis(self.inner.diagnostics_policy.stream_heartbeat_interval_ms);
        let producer = self.inner.diagnostics.snapshot();
        let mut deadline = if producer.exited {
            None
        } else {
            producer
                .received_at
                .and_then(|received| {
                    received.checked_add(Duration::from_millis(
                        self.inner.diagnostics_policy.report_expiry_ms,
                    ))
                })
                .filter(|deadline| *deadline > now)
        };
        let Some(_received_at) = producer.received_at else {
            return heartbeat;
        };
        let Some(report) = producer.report else {
            return heartbeat;
        };
        for stage in &report.stages {
            if stage.status != StageStatus::Healthy || !stage.in_flight {
                continue;
            }
            let (Some(deadline_ms), Some(work_started_age_ms)) =
                (stage.deadline_ms, stage.work_started_age_ms)
            else {
                continue;
            };
            let report_age_ms = producer
                .received_at
                .and_then(|received| now.checked_duration_since(received))
                .map(duration_millis)
                .unwrap_or_default();
            if let Some(work_deadline) = now
                .checked_add(Duration::from_millis(
                    deadline_ms.saturating_sub(work_started_age_ms.saturating_add(report_age_ms)),
                ))
                .filter(|deadline| *deadline > now)
            {
                deadline =
                    Some(deadline.map_or(work_deadline, |current| current.min(work_deadline)));
            }
        }
        deadline
            .map(|deadline| deadline.saturating_duration_since(now).min(heartbeat))
            .unwrap_or(heartbeat)
    }

    pub fn router(&self) -> Router {
        let scene_router = Router::new()
            .route("/api/v1/health/live", get(health_live))
            .route("/api/v1/health/ready", get(health_ready))
            .route("/api/v1/setup", get(setup_status))
            .route("/api/v1/setup/cameras", get(setup_cameras))
            .route(
                "/api/v1/setup/source",
                put(setup_source).layer(DefaultBodyLimit::max(4096)),
            )
            .route(
                "/api/v1/setup/calibration/frame",
                get(setup_calibration_frame),
            )
            .route(
                "/api/v1/setup/calibration",
                post(setup_calibration_apply).layer(DefaultBodyLimit::max(
                    crate::calibration_control::MAXIMUM_CALIBRATION_REQUEST_BYTES,
                )),
            )
            .route(
                "/api/v1/setup/calibration/checkerboard.pdf",
                get(setup_checkerboard_pdf),
            )
            .route(
                "/api/v1/setup/calibration/optics",
                post(setup_fit_optics).layer(DefaultBodyLimit::max(
                    crate::browser_precision_calibration::MAXIMUM_FIT_REQUEST_BYTES,
                )),
            )
            .route(
                "/api/v1/setup/calibration/ball-centers",
                post(setup_fit_ball_centers).layer(DefaultBodyLimit::max(
                    crate::browser_precision_calibration::MAXIMUM_FIT_REQUEST_BYTES,
                )),
            )
            .route(
                "/api/v1/setup/calibration/captures/{mode}",
                get(setup_list_captures).delete(setup_clear_captures),
            )
            .route(
                "/api/v1/setup/calibration/captures/{mode}/{slot}",
                put(setup_store_capture)
                    .delete(setup_remove_capture)
                    .layer(DefaultBodyLimit::max(
                        crate::browser_precision_calibration::MAXIMUM_STILL_REQUEST_BYTES,
                    )),
            )
            .route(
                "/api/v1/setup/calibration/ball-candidate",
                post(setup_ball_candidate).layer(DefaultBodyLimit::max(
                    crate::browser_precision_calibration::MAXIMUM_STILL_REQUEST_BYTES,
                )),
            )
            .route(
                "/api/v1/setup/calibration/ball-placement",
                get(setup_ball_placement),
            )
            .route("/api/v1/websocket/ws", get(websocket_upgrade))
            .route(
                DIAGNOSTICS_WEBSOCKET_PATH,
                get(diagnostics_websocket_upgrade),
            )
            .with_state(self.clone());
        let scene_gateway = self.clone();
        let session_router = GameSessionHttpState::new(
            self.inner.game_session_store.clone(),
            move || {
                scene_gateway
                    .snapshot_now()
                    .map(|state| http_scene_response(&state))
            },
            Arc::clone(&self.inner.configuration_slots),
        )
        .router();
        let router = scene_router
            .merge(session_router)
            .merge(crate::configuration_http::router(
                self.inner.configuration_authority.clone(),
                Arc::clone(&self.inner.configuration_slots),
                Arc::clone(&self.inner.configuration_mutation_lock),
            ));
        match self.inner.config.web_root.as_ref() {
            Some(root) => router
                .fallback(crate::web_static::serve)
                .layer(Extension(crate::web_static::WebRoot(root.clone()))),
            None => router,
        }
    }

    fn monotonic_ns(&self) -> u128 {
        self.inner.started_at.elapsed().as_nanos()
    }

    fn snapshot_now(&self) -> Option<Arc<PublishedState>> {
        let mut state = lock_unpoisoned(&self.inner.state);
        // Sample both clocks only after acquiring the lifecycle lock. A blocked
        // reader must not carry a timestamp older than an ingest that won the lock.
        let now_ns = self.monotonic_ns();
        let snapshot_at_wall = Utc::now();
        let current = state.lifecycle.current(now_ns).ok()??;
        let published_at = state.updated_at?;
        let signature = (
            current.epoch,
            current.updated_monotonic_ns,
            current.freshness,
        );
        let changed = state.published_signature != Some(signature);
        if changed {
            state.publication_sequence = state.publication_sequence.saturating_add(1);
            state.published_signature = Some(signature);
        }
        let candidate = Arc::new(PublishedState {
            current: Arc::new(current),
            published_at,
            snapshot_at_wall,
            snapshot_at_monotonic_ns: now_ns,
            published_at_monotonic_ns: now_ns,
            publication_sequence: state.publication_sequence,
        });
        if changed {
            // The lifecycle lock also serializes watch publication. Otherwise an older
            // snapshot could be delayed after unlock and overwrite a newer state.
            self.inner.latest.send_replace(Some(candidate.clone()));
        }
        Some(candidate)
    }

    fn refresh_freshness(&self) {
        let _ = self.snapshot_now();
    }

    fn freshness_delay(&self, published: &PublishedState) -> Option<Duration> {
        let threshold = match published.current.freshness {
            StateFreshness::Fresh => self.inner.config.stale_after.as_nanos(),
            StateFreshness::Stale => self.inner.config.expire_after.as_nanos(),
            StateFreshness::Expired => return None,
        };
        let _received = published.current.localization_received_monotonic_ns?;
        let age = published
            .current
            .localization_age_ns
            .unwrap_or_default()
            .saturating_add(
                self.monotonic_ns()
                    .saturating_sub(published.snapshot_at_monotonic_ns),
            );
        let remaining = threshold.saturating_sub(age);
        Some(Duration::from_nanos(
            u64::try_from(remaining.max(1)).unwrap_or(u64::MAX),
        ))
    }

    #[cfg(test)]
    fn accept_scene(&self, snapshot: SceneSnapshotV1) -> Result<(), String> {
        self.accept_scene_with_capture_age(snapshot, 0, HashMap::new(), None, None)
    }

    fn accept_scene_with_capture_age(
        &self,
        snapshot: SceneSnapshotV1,
        localization_age_offset_ns: u128,
        ball_age_offsets_ns: HashMap<String, u128>,
        table: Option<Value>,
        game_type: Option<String>,
    ) -> Result<(), String> {
        self.accept_scene_with_capture_age_and_perception(
            snapshot,
            localization_age_offset_ns,
            ball_age_offsets_ns,
            table,
            game_type,
            None,
        )
    }

    /// Ingest a scene and its matching perception envelope as one gateway
    /// state transition. Publishing between these updates would expose new
    /// measured centers with the previous frame's trust and guidance flags.
    fn accept_scene_with_capture_age_and_perception(
        &self,
        snapshot: SceneSnapshotV1,
        localization_age_offset_ns: u128,
        ball_age_offsets_ns: HashMap<String, u128>,
        table: Option<Value>,
        game_type: Option<String>,
        perception_update: Option<PerceptionDeliveryUpdate<'_>>,
    ) -> Result<(), String> {
        let mut perception_status = perception_update
            .as_ref()
            .map(|update| serde_json::to_value(update.perception))
            .transpose()
            .map_err(|error| format!("could not encode perception status: {error}"))?;
        let mut state = lock_unpoisoned(&self.inner.state);
        if let Some(update) = perception_update.as_ref() {
            if !state
                .producer_identity
                .as_ref()
                .is_some_and(|(engine, session)| {
                    engine == update.engine_instance_id && session == update.source_session_id
                })
            {
                return Err("perception delivery source is not the active producer".into());
            }
            if state
                .last_delivery_sequence
                .is_some_and(|previous| update.delivery_sequence <= previous)
            {
                return Err("perception delivery sequence is duplicate or out of order".into());
            }
        }
        if let Some((width_px, height_px)) = state.live_source_geometry {
            if (snapshot.source.width_px, snapshot.source.height_px) != (width_px, height_px) {
                return Err("scene dimensions differ from acknowledged capture geometry".into());
            }
        }
        let now_ns = self.monotonic_ns();
        if let Some(update) = perception_update.as_ref() {
            let current_before_scene = state
                .lifecycle
                .current(now_ns)
                .map_err(|error| error.to_string())?;
            if !layout_basis_is_accepted(
                current_before_scene.as_ref(),
                Some(&snapshot),
                update.perception,
            ) {
                if let Some(status) = perception_status.as_mut() {
                    fail_closed_unaccepted_layout_basis(status);
                }
            }
        }
        let received_at = Utc::now();
        let outcome = state
            .lifecycle
            .ingest_with_source_metadata(
                &snapshot,
                now_ns,
                now_ns,
                SourceStateMetadata {
                    localization_age_offset_ns,
                    ball_age_offsets_ns,
                    table,
                    game_type,
                },
            )
            .map_err(|error| error.to_string())?;
        state.last_transport_sequence = Some(snapshot.transport_sequence);
        if outcome == IngestOutcome::AnnotationIgnored && perception_update.is_none() {
            return Ok(());
        }

        if outcome == IngestOutcome::LocalizationUpdated {
            state.localization_received_at = Some(received_at);
        }
        if let (Some(update), Some(status)) = (perception_update.as_ref(), perception_status) {
            state
                .lifecycle
                .update_perception_status(
                    update.engine_instance_id,
                    update.source_session_id,
                    status,
                    now_ns,
                    now_ns,
                )
                .map_err(|error| error.to_string())?;
            state.last_delivery_sequence = Some(update.delivery_sequence);
        }
        state.updated_at = Some(received_at);
        let current = state
            .lifecycle
            .current(now_ns)
            .map_err(|error| error.to_string())?;
        if let Some(current) = current {
            if current.localization_received_monotonic_ns.is_some() || perception_update.is_some() {
                state.publication_sequence = state.publication_sequence.saturating_add(1);
                state.published_signature = Some((
                    current.epoch,
                    current.updated_monotonic_ns,
                    current.freshness,
                ));
                let published = Arc::new(PublishedState {
                    current: Arc::new(current),
                    published_at: received_at,
                    snapshot_at_wall: received_at,
                    snapshot_at_monotonic_ns: now_ns,
                    published_at_monotonic_ns: now_ns,
                    publication_sequence: state.publication_sequence,
                });
                self.inner.latest.send_replace(Some(published));
                return Ok(());
            }
        }
        Ok(())
    }

    fn resolve_live_capture_settings(&self) -> Result<ResolvedCaptureSettings, String> {
        let authority = self
            .inner
            .configuration_authority
            .as_ref()
            .ok_or_else(|| "live-v3 ingress requires an explicit --config path".to_owned())?;
        let request = serde_json::to_vec(&json!({
            "protocol_version": 1,
            "operation": "capture_settings",
            "data": {"schema_version": 3},
        }))
        .map_err(|error| format!("could not encode capture-settings request: {error}"))?;
        let (response_bytes, process_ok) = authority.handle_json(&request);
        let response: Value = serde_json::from_slice(&response_bytes)
            .map_err(|error| format!("configuration authority returned invalid JSON: {error}"))?;
        if !process_ok || response.get("ok").and_then(Value::as_bool) != Some(true) {
            let message = response
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("configuration authority rejected capture settings");
            let detail = response
                .get("validation_errors")
                .and_then(Value::as_array)
                .and_then(|errors| errors.first())
                .and_then(|error| {
                    Some((
                        error.get("field")?.as_str()?,
                        error.get("message")?.as_str()?,
                    ))
                })
                .map(|(field, detail)| format!(" ({field}: {detail})"))
                .unwrap_or_default();
            return Err(format!(
                "live capture configuration unavailable: {message}{detail}"
            ));
        }
        let capture_settings = response
            .get("capture_settings")
            .ok_or_else(|| "configuration authority omitted capture settings".to_owned())?;
        let exact_json = serde_json::to_string(capture_settings)
            .map_err(|error| format!("could not encode active capture settings: {error}"))?;
        let resolved = parse_capture_settings_json(&exact_json)
            .map_err(|error| format!("active capture settings are invalid: {error}"))?;
        if resolved.schema_version != LIVE_PROTOCOL_VERSION {
            return Err("configuration authority did not resolve capture settings v3".into());
        }
        Ok(resolved)
    }

    fn acknowledge_live_producer<W: Write>(
        &self,
        writer: &mut W,
        handshake: &ValidatedLiveHandshake,
    ) -> Result<(), String> {
        let engine = handshake.identity.engine_instance_id.clone();
        let session = handshake.identity.source_session_id.clone();
        let watermark = self.producer_watermark(&engine, &session);
        let ready = json!({
            "type": "ready",
            "protocol_version": LIVE_PROTOCOL_VERSION,
            "configuration_revision": handshake.acknowledgement.configuration_revision,
            "request_sha256": handshake.acknowledgement.request_sha256,
            "last_transport_sequence": watermark,
        });
        write_framed_message(
            writer,
            &serde_json::to_vec(&ready).map_err(|error| error.to_string())?,
            self.inner.config.maximum_scene_bytes,
        )
        .map_err(|error| error.to_string())?;
        self.activate_producer_epoch(engine, session)?;
        let geometry = handshake.compatibility.output_frame_geometry();
        let mut state = lock_unpoisoned(&self.inner.state);
        state.live_source_geometry = Some((geometry.width_px(), geometry.height_px()));
        state.live_configuration_revision = Some(handshake.acknowledgement.configuration_revision);
        Ok(())
    }

    fn producer_watermark(&self, engine: &str, session: &str) -> Option<u64> {
        let state = lock_unpoisoned(&self.inner.state);
        match state.producer_identity.as_ref() {
            Some((active_engine, active_session))
                if active_engine == engine && active_session == session =>
            {
                state.last_transport_sequence
            }
            _ => None,
        }
    }

    fn activate_producer_epoch(&self, engine: String, session: String) -> Result<(), String> {
        let mut state = lock_unpoisoned(&self.inner.state);
        let identity = (engine, session);
        if state.producer_identity.as_ref() != Some(&identity) {
            state
                .lifecycle
                .activate_source(identity.0.clone(), identity.1.clone())
                .map_err(|error| error.to_string())?;
            state.producer_identity = Some(identity);
            state.last_transport_sequence = None;
            state.last_delivery_sequence = None;
            state.localization_received_at = None;
            state.updated_at = None;
            state.live_source_geometry = None;
            state.live_configuration_revision = None;
            state.published_signature = None;
            // Commit epoch state and clear public state atomically with respect to
            // accept_scene/snapshot_now and all watch publications.
            self.inner.latest.send_replace(None);
        }
        Ok(())
    }

    fn handle_producer(&self, mut stream: std::net::TcpStream) -> Result<(), String> {
        let _ = stream.set_nodelay(true);
        stream
            .set_nonblocking(false)
            .map_err(|error| format!("could not prepare scene stream: {error}"))?;
        stream
            .set_read_timeout(Some(self.inner.config.tcp_read_timeout))
            .map_err(|error| format!("could not set scene read timeout: {error}"))?;
        stream
            .set_write_timeout(Some(self.inner.config.tcp_read_timeout))
            .map_err(|error| format!("could not set scene write timeout: {error}"))?;
        let hello_bytes = read_framed_message(&mut stream, self.inner.config.maximum_scene_bytes)
            .map_err(|error| error.to_string())?;
        let capture_settings = self.resolve_live_capture_settings()?;
        let handshake = validate_live_v3_hello(&hello_bytes, &capture_settings)
            .map_err(|error| error.to_string())?;
        self.acknowledge_live_producer(&mut stream, &handshake)?;

        loop {
            let bytes = read_framed_message(&mut stream, self.inner.config.maximum_scene_bytes)
                .map_err(|error| error.to_string())?;
            self.accept_live_delivery(&bytes, &handshake)?;
        }
    }

    fn accept_live_delivery(
        &self,
        bytes: &[u8],
        handshake: &ValidatedLiveHandshake,
    ) -> Result<(), String> {
        let header: PublicStateDeliveryHeader = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid public-state delivery header: {error}"))?;
        if header.delivery_type != "billiards_public_state_delivery" {
            return Err("unsupported public-state delivery contract".into());
        }
        if header.delivery_schema_version == 2 {
            return self.accept_live_delivery_v2(bytes, handshake);
        }
        let delivery: PublicStateDeliveryInput = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid public-state delivery v1: {error}"))?;
        if delivery.delivery_type != "billiards_public_state_delivery"
            || delivery.delivery_schema_version != 1
        {
            return Err("unsupported public-state delivery contract".into());
        }
        let snapshot = delivery.source_snapshot;
        snapshot
            .validate()
            .map_err(|error| format!("invalid source scene snapshot: {error}"))?;
        if snapshot.engine_instance_id != handshake.identity.engine_instance_id
            || snapshot.source.session_id != handshake.identity.source_session_id
        {
            return Err("scene identity differs from the acknowledged live source".into());
        }
        validate_live_public_state(&snapshot, delivery.current_state.as_ref())?;

        // Keep the validated raw scene as the authority for ball locations.
        // Public ball fields are rebuilt by the same Rust lifecycle adapter;
        // carry only non-scene metadata and elapsed age durations across the
        // process boundary. Producer monotonic instants are never reused.
        let (localization_age_offset_ns, ball_age_offsets_ns, table, game_type) =
            match delivery.current_state {
                Some(current) => (
                    current.localization_age_ns.unwrap_or_default(),
                    current.ball_age_ns,
                    current.payload.table,
                    current.payload.game_type,
                ),
                None => (0, HashMap::new(), None, None),
            };
        self.accept_scene_with_capture_age(
            snapshot,
            localization_age_offset_ns,
            ball_age_offsets_ns,
            table,
            game_type,
        )
    }

    fn accept_live_delivery_v2(
        &self,
        bytes: &[u8],
        handshake: &ValidatedLiveHandshake,
    ) -> Result<(), String> {
        let delivery: PublicStateDeliveryV2Input =
            serde_json::from_slice(bytes).map_err(|error| {
                format!("unsupported public-state delivery contract or invalid v2 payload: {error}")
            })?;
        if delivery.delivery_type != "billiards_public_state_delivery"
            || delivery.delivery_schema_version != 2
            || delivery.delivery_sequence == 0
            || delivery.perception.event_sequence != delivery.delivery_sequence
        {
            return Err("unsupported or inconsistent public-state delivery v2".into());
        }
        if delivery.engine_instance_id != handshake.identity.engine_instance_id
            || delivery.source_session_id != handshake.identity.source_session_id
        {
            return Err(
                "perception delivery identity differs from the acknowledged live source".into(),
            );
        }
        validate_perception_state(&delivery.perception)?;
        if let Some(snapshot) = delivery.source_snapshot {
            snapshot
                .validate()
                .map_err(|error| format!("invalid source scene snapshot: {error}"))?;
            if snapshot.engine_instance_id != handshake.identity.engine_instance_id
                || snapshot.source.session_id != handshake.identity.source_session_id
            {
                return Err("scene identity differs from the acknowledged live source".into());
            }
            validate_live_public_state(&snapshot, delivery.current_state.as_ref())?;
            let (localization_age_offset_ns, ball_age_offsets_ns, table, game_type) =
                match delivery.current_state {
                    Some(current) => (
                        current.localization_age_ns.unwrap_or_default(),
                        current.ball_age_ns,
                        current.payload.table,
                        current.payload.game_type,
                    ),
                    None => (0, HashMap::new(), None, None),
                };
            return self.accept_scene_with_capture_age_and_perception(
                snapshot,
                localization_age_offset_ns,
                ball_age_offsets_ns,
                table,
                game_type,
                Some(PerceptionDeliveryUpdate {
                    delivery_sequence: delivery.delivery_sequence,
                    engine_instance_id: &delivery.engine_instance_id,
                    source_session_id: &delivery.source_session_id,
                    perception: &delivery.perception,
                }),
            );
        } else if let Some(current) = delivery.current_state.as_ref() {
            if current.epoch == 0
                || current.payload.balls.len() > StateLifecycleConfig::default().maximum_balls
                || !current.payload.observation_status.is_object()
                || current.payload.observation_status["perception"]["event_sequence"].as_u64()
                    != Some(delivery.delivery_sequence)
            {
                return Err("phase-only delivery contains an invalid current state".into());
            }
        }

        self.accept_perception_state(
            delivery.delivery_sequence,
            &delivery.engine_instance_id,
            &delivery.source_session_id,
            delivery.perception,
        )
    }

    fn accept_perception_state(
        &self,
        delivery_sequence: u64,
        engine_instance_id: &str,
        source_session_id: &str,
        perception: PerceptionStateV1,
    ) -> Result<(), String> {
        let mut state = lock_unpoisoned(&self.inner.state);
        if !state
            .producer_identity
            .as_ref()
            .is_some_and(|(engine, session)| {
                engine == engine_instance_id && session == source_session_id
            })
        {
            return Err("perception delivery source is not the active producer".into());
        }
        let sequence_is_stale = state
            .last_delivery_sequence
            .is_some_and(|previous| delivery_sequence <= previous);
        if sequence_is_stale {
            return Err("perception delivery sequence is duplicate or out of order".into());
        }
        let now_ns = self.monotonic_ns();
        let received_at = Utc::now();
        let mut status = serde_json::to_value(&perception)
            .map_err(|error| format!("could not encode perception status: {error}"))?;
        let current_before_update = state
            .lifecycle
            .current(now_ns)
            .map_err(|error| error.to_string())?;
        if !layout_basis_is_accepted(current_before_update.as_ref(), None, &perception) {
            fail_closed_unaccepted_layout_basis(&mut status);
        }
        state
            .lifecycle
            .update_perception_status(
                engine_instance_id,
                source_session_id,
                status,
                now_ns,
                now_ns,
            )
            .map_err(|error| error.to_string())?;
        state.last_delivery_sequence = Some(delivery_sequence);
        state.updated_at = Some(received_at);
        let current = state
            .lifecycle
            .current(now_ns)
            .map_err(|error| error.to_string())?;
        if let Some(current) = current {
            // Publish phase-only states even before the first localized scene.
            // Their payload contains no ball positions and marks them
            // unavailable; the cue can still be drawn from its own fresh
            // evidence clock.
            state.publication_sequence = state.publication_sequence.saturating_add(1);
            state.published_signature = Some((
                current.epoch,
                current.updated_monotonic_ns,
                current.freshness,
            ));
            self.inner
                .latest
                .send_replace(Some(Arc::new(PublishedState {
                    current: Arc::new(current),
                    published_at: received_at,
                    snapshot_at_wall: received_at,
                    snapshot_at_monotonic_ns: now_ns,
                    published_at_monotonic_ns: now_ns,
                    publication_sequence: state.publication_sequence,
                })));
        }
        Ok(())
    }

    fn shutdown(&self) {
        self.inner.shutdown.send_replace(true);
        {
            let _state = lock_unpoisoned(&self.inner.state);
            self.inner.latest.send_replace(None);
        }
        if let Some(stream) = lock_unpoisoned(&self.inner.producer_control_stream).as_ref() {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn stage_assessment(
    stage: DiagnosticsStageName,
    status: StageStatus,
    reason_code: &str,
    source: StageEvidenceSource,
    detail: Option<&str>,
    last_progress_age_ms: Option<u64>,
    deadline_ms: Option<u64>,
) -> BrowserDiagnosticsStage {
    BrowserDiagnosticsStage {
        stage,
        status,
        reason_code: reason_code.to_owned(),
        source,
        detail: detail.map(str::to_owned),
        last_progress_age_ms,
        in_flight: false,
        work_started_age_ms: None,
        expected_jobs: None,
        completed_jobs: None,
        failed_jobs: None,
        scheduler_state: None,
        active_job_kind: None,
        last_completion_age_ms: None,
        last_failure_code: None,
        last_failure_age_ms: None,
        deadline_ms,
    }
}

fn initial_browser_diagnostics_snapshot(
    gateway_instance_id: &str,
    policy: &DiagnosticsConfig,
) -> BrowserDiagnosticsSnapshot {
    let mut stages = vec![stage_assessment(
        DiagnosticsStageName::GatewayService,
        StageStatus::Healthy,
        "gateway_running",
        StageEvidenceSource::GatewayRuntime,
        None,
        Some(0),
        None,
    )];
    for (name, status, reason, source) in [
        (
            DiagnosticsStageName::SourceLifecycle,
            StageStatus::Unknown,
            "producer_evidence_absent",
            StageEvidenceSource::SupervisorRegistry,
        ),
        (
            DiagnosticsStageName::CaptureProgress,
            StageStatus::Unknown,
            "producer_evidence_absent",
            StageEvidenceSource::ProducerReport,
        ),
        (
            DiagnosticsStageName::ModelInitialization,
            StageStatus::Unknown,
            "producer_evidence_absent",
            StageEvidenceSource::ProducerReport,
        ),
        (
            DiagnosticsStageName::ScheduledPerceptionWork,
            StageStatus::Unknown,
            "producer_evidence_absent",
            StageEvidenceSource::ProducerReport,
        ),
        (
            DiagnosticsStageName::SceneDelivery,
            StageStatus::Starting,
            "scene_not_received",
            StageEvidenceSource::SceneLifecycle,
        ),
        (
            DiagnosticsStageName::Localization,
            StageStatus::Unknown,
            "localization_not_received",
            StageEvidenceSource::SceneLifecycle,
        ),
    ] {
        stages.push(stage_assessment(
            name, status, reason, source, None, None, None,
        ));
    }
    BrowserDiagnosticsSnapshot {
        schema_version: BROWSER_DIAGNOSTICS_SCHEMA_VERSION,
        gateway: GatewayRuntimeSnapshot {
            instance_id: gateway_instance_id.to_owned(),
            revision: 1,
            assessed_at_utc: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
            uptime_ms: 0,
            build: BuildInfo::current(),
        },
        producer: ProducerReportSnapshot {
            state: ProducerReportState::Unknown,
            generation_id: None,
            report_sequence: None,
            report_age_ms: None,
            expires_in_ms: None,
            metadata: None,
            exit_reason: None,
            last_known: None,
        },
        stream: BrowserDiagnosticsStreamPolicy {
            heartbeat_interval_ms: policy.stream_heartbeat_interval_ms,
            stale_after_ms: policy.stream_stale_after_ms,
        },
        stages,
    }
}

fn configured_game_session_store(
    authority: &ConfigurationAuthority,
    project_root: &std::path::Path,
) -> Option<GameSessionStore> {
    let configuration = match authority.effective_configuration() {
        Ok(configuration) => configuration,
        Err(error) => {
            eprintln!("game_session_store_unavailable=configuration_read_failed:{error}");
            return None;
        }
    };
    let Some(settings) = configuration.get("game_session") else {
        eprintln!("game_session_store_unavailable=configuration_section_missing");
        return None;
    };
    let Some(store_path) = settings.get("store_path").and_then(Value::as_str) else {
        eprintln!("game_session_store_unavailable=store_path_missing");
        return None;
    };
    let Some(maximum_store_bytes) = settings
        .get("maximum_store_file_bytes")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
    else {
        eprintln!("game_session_store_unavailable=maximum_store_file_bytes_invalid");
        return None;
    };
    let Some(lock_timeout_ms) = settings.get("lock_timeout_ms").and_then(Value::as_u64) else {
        eprintln!("game_session_store_unavailable=lock_timeout_ms_invalid");
        return None;
    };
    if store_path.trim().is_empty() || store_path.len() > 4096 {
        eprintln!("game_session_store_unavailable=store_path_invalid");
        return None;
    }
    let resolved_path = match resolve_store_path_under_root(PathBuf::from(store_path), project_root)
    {
        Ok(path) => path,
        Err(error) => {
            eprintln!("game_session_store_unavailable=store_path_outside_project_root:{error}");
            return None;
        }
    };
    match GameSessionStore::new_confined(
        resolved_path,
        project_root,
        Duration::from_millis(lock_timeout_ms),
        maximum_store_bytes,
    ) {
        Ok(store) => Some(store),
        Err(error) => {
            eprintln!("game_session_store_unavailable=invalid_settings:{error}");
            None
        }
    }
}

fn resolve_store_path_under_root(
    path: PathBuf,
    project_root: &std::path::Path,
) -> Result<PathBuf, String> {
    let root = std::fs::canonicalize(project_root)
        .map_err(|error| format!("cannot resolve project root: {error}"))?;
    if !root.is_dir() {
        return Err("project root is not a directory".into());
    }
    let candidate = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    let mut normalized_candidate = PathBuf::new();
    for component in candidate.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized_candidate.pop();
            }
            other => normalized_candidate.push(other.as_os_str()),
        }
    }
    if normalized_candidate.file_name().is_none() {
        return Err("store path must name a file".into());
    }

    // Resolve the nearest existing ancestor so symlinks cannot redirect a
    // not-yet-created store outside the deployment-owned data tree.
    let mut existing = normalized_candidate.clone();
    let mut missing = Vec::new();
    while !existing.exists() {
        let name = existing
            .file_name()
            .ok_or_else(|| "store path has no existing ancestor".to_owned())?
            .to_os_string();
        missing.push(name);
        if !existing.pop() {
            return Err("store path has no existing ancestor".into());
        }
    }
    let mut resolved = std::fs::canonicalize(&existing)
        .map_err(|error| format!("cannot resolve store path ancestor: {error}"))?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    if !resolved.starts_with(&root) {
        return Err("store path must remain under the deployment project root".into());
    }
    Ok(resolved)
}

struct ActiveProducerGuard(Arc<GatewayInner>);

impl Drop for ActiveProducerGuard {
    fn drop(&mut self) {
        lock_unpoisoned(&self.0.producer_control_stream).take();
        lock_unpoisoned(&self.0.state).live_configuration_revision = None;
        self.0.producer_connected.store(false, Ordering::Release);
        self.0.producer_connection_updates.send_replace(false);
    }
}

#[derive(Debug, Deserialize)]
struct PublicStateDeliveryHeader {
    delivery_type: String,
    delivery_schema_version: u8,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicStateDeliveryInput {
    delivery_type: String,
    delivery_schema_version: u8,
    source_snapshot: SceneSnapshotV1,
    #[serde(deserialize_with = "required_delivery_option")]
    current_state: Option<CurrentSceneState>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicStateDeliveryV2Input {
    delivery_type: String,
    delivery_schema_version: u8,
    delivery_sequence: u64,
    engine_instance_id: String,
    source_session_id: String,
    #[serde(deserialize_with = "required_delivery_option")]
    source_snapshot: Option<SceneSnapshotV1>,
    #[serde(deserialize_with = "required_delivery_option")]
    current_state: Option<CurrentSceneState>,
    perception: PerceptionStateV1,
}

struct PerceptionDeliveryUpdate<'a> {
    delivery_sequence: u64,
    engine_instance_id: &'a str,
    source_session_id: &'a str,
    perception: &'a PerceptionStateV1,
}

fn required_delivery_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn validate_live_public_state(
    snapshot: &SceneSnapshotV1,
    current_state: Option<&CurrentSceneState>,
) -> Result<(), String> {
    let is_localization = snapshot.update_kind == crate::UpdateKind::Observation
        && snapshot.outcome == crate::Outcome::Success;
    if is_localization && current_state.is_none() {
        return Err("successful live observation omitted current state".into());
    }
    let Some(current) = current_state else {
        return Ok(());
    };
    if current.epoch == 0 {
        return Err("live public-state epoch must be positive".into());
    }
    if current.payload.balls.len() > StateLifecycleConfig::default().maximum_balls {
        return Err("live public state exceeds the configured ball limit".into());
    }
    if !current.payload.observation_status.is_object() {
        return Err("live observation_status must be an object".into());
    }
    for ball in &current.payload.balls {
        validate_live_public_ball(ball, snapshot)?;
    }
    if is_localization {
        let status = &current.payload.observation_status;
        if let Some(sequence) = status.get("transport_sequence").and_then(Value::as_u64) {
            if sequence != snapshot.transport_sequence {
                return Err("public-state sequence differs from its source snapshot".into());
            }
        } else if current.freshness == StateFreshness::Fresh {
            return Err("fresh public state omitted its source sequence".into());
        }
        if current.freshness == StateFreshness::Fresh
            && (current.localization_age_ns.is_none()
                || status.get("snapshot_revision").and_then(Value::as_u64)
                    != Some(snapshot.snapshot_revision)
                || status.get("source_session_id").and_then(Value::as_str)
                    != Some(snapshot.source.session_id.as_str())
                || status.get("source_frame_id").and_then(Value::as_str)
                    != Some(snapshot.source.frame_id.as_str())
                || status.get("engine_instance_id").and_then(Value::as_str)
                    != Some(snapshot.engine_instance_id.as_str()))
        {
            return Err("fresh public-state identity differs from its source snapshot".into());
        }
    }
    if current.freshness == StateFreshness::Fresh
        && current.localization_received_monotonic_ns.is_none()
    {
        return Err("fresh public state omitted its localization receipt time".into());
    }
    Ok(())
}

fn validate_perception_state(state: &PerceptionStateV1) -> Result<(), String> {
    if !matches!(
        state.phase.as_str(),
        "recovering" | "pre_shot" | "aiming" | "shot_underway"
    ) || state.trust_reason.is_empty()
        || state.trust_reason.len() > 128
        || state.motion_evidence_max_age_ms == 0
        || state.cue_evidence_max_age_ms == 0
        || state.layout_verification_max_age_ms == 0
    {
        return Err("perception delivery contains invalid phase or evidence limits".into());
    }
    if state.guidance_valid {
        let cue_is_fresh = state
            .cue
            .as_ref()
            .is_some_and(|cue| cue.observation_age_ms <= state.cue_evidence_max_age_ms)
            && state
                .cue_evidence_age_ms
                .is_some_and(|age| age <= state.cue_evidence_max_age_ms);
        if !state.layout_trusted
            || !state
                .latest_motion_check_age_ms
                .is_some_and(|age| age <= state.motion_evidence_max_age_ms)
            || !cue_is_fresh
        {
            return Err(
                "guidance requires trusted layout, current whole-table motion, and timely cue evidence"
                    .into(),
            );
        }
    }
    if state.decision_calibration_revision.is_some()
        != state.decision_rectification_revision.is_some()
        || (state.decision_frame_id.is_some()
            && (state.image_width_px.is_none() || state.image_height_px.is_none()))
        || state
            .layout_basis_frame_id
            .as_ref()
            .is_some_and(|frame_id| frame_id.is_empty() || frame_id.len() > 128)
        || state
            .decision_frame_id
            .as_ref()
            .is_some_and(|frame_id| frame_id.is_empty() || frame_id.len() > 128)
    {
        return Err("perception decision frame provenance is incomplete or invalid".into());
    }
    if state.table_visibility.as_ref().is_some_and(|visibility| {
        visibility.observation_frame_id.is_empty()
            || visibility.observation_frame_id.len() > 128
            || visibility.reason.is_empty()
            || visibility.reason.len() > 128
    }) {
        return Err("perception table-visibility evidence is invalid".into());
    }
    if let Some(projection) = state.table_projection.as_ref() {
        let image_space = match projection.image_space.as_str() {
            "raw" => crate::ImageSpace::Raw,
            "rectified" => crate::ImageSpace::Rectified,
            _ => return Err("perception table projection has invalid image space".into()),
        };
        let calibration = crate::calibration::TableCalibrationV1 {
            schema_version: crate::calibration::TABLE_CALIBRATION_SCHEMA_VERSION,
            calibration_revision: projection.calibration_revision,
            camera_profile_id: projection.camera_profile_id.clone(),
            rectification_revision: projection.rectification_revision,
            image_space,
            source_width_px: projection.source_width_px,
            source_height_px: projection.source_height_px,
            coordinate_frame:
                crate::calibration::TableCoordinateFrame::CenterOriginXCorner0To1YCorner0To3,
            table_width_mm: projection.table_width_mm,
            table_height_mm: projection.table_height_mm,
            pixel_to_table_mm: projection.pixel_to_table_mm,
        };
        if projection.direction != "source_image_px_to_table_mm"
            || projection.unit != "millimeters"
            || projection.origin != "playing_surface_center"
            || projection.x_axis != "corner_0_to_1"
            || projection.y_axis != "corner_0_to_3"
            || state
                .image_width_px
                .is_some_and(|width| width != projection.source_width_px)
            || state
                .image_height_px
                .is_some_and(|height| height != projection.source_height_px)
            || state
                .decision_calibration_revision
                .is_some_and(|revision| revision != projection.calibration_revision)
            || state
                .decision_rectification_revision
                .is_some_and(|revision| revision != projection.rectification_revision)
            || calibration.validate().is_err()
        {
            return Err(
                "perception table projection is invalid or does not match the decision context"
                    .into(),
            );
        }
    }
    if state.cue.as_ref().is_some_and(|cue| {
        cue.source_frame_id
            .as_ref()
            .is_some_and(|frame_id| frame_id.is_empty() || frame_id.len() > 128)
    }) {
        return Err("perception cue evidence has an invalid source frame identity".into());
    }
    if state.empty_layout_verified
        && (!state.layout_trusted
            || !state
                .latest_motion_check_age_ms
                .is_some_and(|age| age <= state.motion_evidence_max_age_ms))
    {
        return Err(
            "an empty layout requires trusted layout, motion, and observation evidence".into(),
        );
    }
    if state.image_width_px.is_some() != state.image_height_px.is_some()
        || state.image_width_px == Some(0)
        || state.image_height_px == Some(0)
    {
        return Err("perception delivery contains incomplete image dimensions".into());
    }
    if let Some(cue) = state.cue.as_ref() {
        if !cue.confidence.is_finite()
            || !(0.0..=1.0).contains(&cue.confidence)
            || cue
                .shaft_start_px
                .iter()
                .chain(cue.shaft_end_px.iter())
                .chain(cue.axis_unit.iter())
                .any(|coordinate| !coordinate.is_finite())
            || cue
                .signed_direction
                .as_ref()
                .is_some_and(|direction| direction != "toward_start" && direction != "toward_end")
        {
            return Err("perception delivery contains invalid cue geometry".into());
        }
        let (Some(width), Some(height)) = (state.image_width_px, state.image_height_px) else {
            return Err("cue geometry omitted its source image dimensions".into());
        };
        if cue
            .shaft_start_px
            .iter()
            .zip([f64::from(width), f64::from(height)])
            .chain(
                cue.shaft_end_px
                    .iter()
                    .zip([f64::from(width), f64::from(height)]),
            )
            .any(|(coordinate, limit)| *coordinate < 0.0 || *coordinate > limit)
        {
            return Err("cue geometry lies outside its source image".into());
        }
    }
    Ok(())
}

/// Trust can be restored only when its exact verified layout basis has already
/// reached the gateway, either as the scene currently being ingested or as an
/// already accepted trusted layout. This prevents a later phase-only update
/// from blessing older centers when its clean scene expired during reconnect.
fn layout_basis_is_accepted(
    current: Option<&CurrentSceneState>,
    incoming_scene: Option<&SceneSnapshotV1>,
    perception: &PerceptionStateV1,
) -> bool {
    if !perception.layout_trusted {
        return true;
    }
    let Some(basis_frame_id) = perception.layout_basis_frame_id.as_deref() else {
        return false;
    };
    if incoming_scene.is_some_and(|snapshot| {
        snapshot.update_kind == crate::UpdateKind::Observation
            && snapshot.outcome == crate::Outcome::Success
            && snapshot.source.frame_id == basis_frame_id
    }) {
        return true;
    }
    let Some(current) = current else {
        return false;
    };
    let status = &current.payload.observation_status;
    let accepted_source_frame_id = status
        .get("source_frame_id")
        .or_else(|| status.pointer("/observation_identity/frame_id"))
        .and_then(Value::as_str);
    if accepted_source_frame_id == Some(basis_frame_id) {
        return true;
    }
    status
        .pointer("/perception/layout_trusted")
        .and_then(Value::as_bool)
        == Some(true)
        && status
            .pointer("/perception/layout_basis_frame_id")
            .and_then(Value::as_str)
            == Some(basis_frame_id)
}

fn fail_closed_unaccepted_layout_basis(status: &mut Value) {
    status["layout_trusted"] = json!(false);
    status["empty_layout_verified"] = json!(false);
    status["guidance_valid"] = json!(false);
    status["trust_reason"] = json!("layout_basis_scene_not_accepted");
}

fn validate_live_public_ball(
    ball: &crate::public_state::PublicBall,
    snapshot: &SceneSnapshotV1,
) -> Result<(), String> {
    let finite_point = |x: f64, y: f64| x.is_finite() && y.is_finite();
    if ball.id.is_empty()
        || ball.id.len() > MAX_IDENTIFIER_CHARACTERS
        || !finite_point(ball.position.x, ball.position.y)
        || !ball
            .position
            .scale
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !ball.radius.is_finite()
        || ball.radius <= 0.0
        || ball.radius_extent.as_ref().is_some_and(|extent| {
            !extent.x.is_finite()
                || !extent.y.is_finite()
                || extent.x <= 0.0
                || extent.y <= 0.0
                || extent
                    .scale
                    .iter()
                    .any(|scale| !scale.is_finite() || *scale <= 0.0)
        })
        || !ball.confidence.is_finite()
        || !(0.0..=1.0).contains(&ball.confidence)
        || ball
            .last_observation_confidence
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        || ball
            .identity_confidence
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        || ball.radius_unit != "canonical_pixels"
        || ball.radius_source.is_empty()
        || !matches!(
            ball.confidence_kind.as_str(),
            "detector_observation" | "unavailable"
        )
        || !matches!(
            ball.ball_type.as_str(),
            "unknown" | "eight" | "stripe" | "solid"
        )
        || !matches!(ball.identity_status.as_str(), "identified" | "unknown")
        || !matches!(
            ball.position_kind.as_str(),
            "measured" | "filtered" | "extrapolated" | "last_observed"
        )
        || ball
            .number
            .is_some_and(|number| !(1..=15).contains(&number))
        || ball.image_space != crate::ImageSpace::Raw
        || ball.image_width_px == 0
        || ball.image_height_px == 0
    {
        return Err("live public ball contains invalid or unsupported values".into());
    }
    if ball.source_observation.engine_instance_id != snapshot.engine_instance_id
        || ball.source_observation.session_id != snapshot.source.session_id
    {
        return Err("public ball provenance differs from the acknowledged source".into());
    }
    if ball.is_observed
        && (ball.tracking_status != crate::BallStatus::Observed
            || ball.source_observation.frame_id != snapshot.source.frame_id
            || ball.source_observation.timestamp != snapshot.source.timestamp)
    {
        return Err("observed public ball differs from the source frame".into());
    }
    if let Some(outline) = &ball.outline {
        if !ball.is_observed || ball.position_kind != "measured" || outline.scale != [1.0, 1.0] {
            return Err(
                "precise public outline requires a current observed image measurement".into(),
            );
        }
        let source_ball = snapshot
            .balls
            .iter()
            .find(|source| source.track_id == ball.id)
            .ok_or("public outline has no matching source ball")?;
        let measurement = source_ball
            .measurement
            .as_ref()
            .ok_or("public outline has no image measurement")?;
        let original = measurement
            .image_outline_px
            .as_ref()
            .ok_or("public outline is not present in its source measurement")?;
        let config = PublicStateConfig::default();
        if source_ball.status != crate::BallStatus::Observed
            || source_ball.last_observation.frame_id != snapshot.source.frame_id
            || source_ball.last_observation.session_id != snapshot.source.session_id
            || source_ball.last_observation.timestamp != snapshot.source.timestamp
            || outline.points.len() != original.len()
            || !crate::valid_image_outline(original, measurement.image_center_px)
        {
            return Err("public outline differs from the exact current source measurement".into());
        }
        let close = |a: f64, b: f64| {
            a.is_finite()
                && b.is_finite()
                && (a - b).abs() <= f64::EPSILON * (1. + a.abs().max(b.abs())) * 16.
        };
        for (point, source_point) in outline.points.iter().zip(original) {
            let x = source_point.x * f64::from(config.canonical_width_px)
                / f64::from(snapshot.source.width_px);
            let y = source_point.y * f64::from(config.canonical_height_px)
                / f64::from(snapshot.source.height_px);
            if !close(point.x, x) || !close(point.y, y) {
                return Err(
                    "public outline coordinates differ from their source-frame geometry".into(),
                );
            }
        }
    }
    if let Some(position) = &ball.table_position {
        if !finite_point(position.x, position.y)
            || position.unit != "millimeters"
            || position.origin != "playing_surface_center"
        {
            return Err("public ball table position is invalid".into());
        }
    }
    for evidence in &ball.identity_evidence {
        if evidence.source.is_empty()
            || evidence.source.len() > MAX_IDENTIFIER_CHARACTERS
            || evidence
                .confidence
                .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err("public ball identity evidence is invalid".into());
        }
    }
    Ok(())
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Resolve a loopback frame-preview listener that the gateway can bridge to
/// the browser as identity-preserving MJPEG. Port zero remains a valid
/// capture-only mode, but its child-selected ephemeral address cannot be
/// resolved by the gateway.
fn configured_frame_preview_stream(
    authority: Option<&ConfigurationAuthority>,
) -> Result<Option<FramePreviewStreamSettings>, String> {
    let Some(authority) = authority else {
        return Ok(None);
    };
    let configuration = authority.effective_configuration()?;
    let live = configuration
        .pointer("/perception/rust_live")
        .and_then(Value::as_object)
        .ok_or_else(|| "perception.rust_live is missing or invalid".to_owned())?;
    if live.get("frame_preview_enabled").and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    let bind_address = live
        .get("frame_preview_bind_address")
        .and_then(Value::as_str)
        .ok_or_else(|| "frame-preview bind address is missing".to_owned())?
        .parse::<IpAddr>()
        .map_err(|error| format!("frame-preview bind address is invalid: {error}"))?;
    if !bind_address.is_loopback() {
        return Err("frame-preview stream address must be loopback".to_owned());
    }
    let port = live
        .get("frame_preview_port")
        .and_then(Value::as_u64)
        .and_then(|port| u16::try_from(port).ok())
        .ok_or_else(|| "frame-preview port is invalid".to_owned())?;
    if port == 0 {
        return Ok(None);
    }
    let positive_size = |key: &str| {
        live.get(key)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("perception.rust_live.{key} must be a positive integer"))
    };
    let request_timeout_ms = live
        .get("frame_preview_request_timeout_ms")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            "perception.rust_live.frame_preview_request_timeout_ms must be positive".to_owned()
        })?;
    let maximum_header_bytes = positive_size("frame_preview_max_header_bytes")?;
    let calibration = if live
        .get("frame_preview_calibration_control_enabled")
        .and_then(Value::as_bool)
        == Some(true)
    {
        Some(CalibrationPreviewLimits {
            maximum_request_bytes: positive_size("frame_preview_calibration_max_request_bytes")?,
            maximum_response_bytes: maximum_header_bytes.saturating_add(positive_size(
                "frame_preview_calibration_store_max_file_bytes",
            )?),
        })
    } else {
        None
    };
    Ok(Some(FramePreviewStreamSettings {
        address: SocketAddr::new(bind_address, port),
        request_timeout: Duration::from_millis(request_timeout_ms),
        maximum_header_bytes,
        maximum_jpeg_bytes: positive_size("frame_preview_max_jpeg_bytes")?,
        calibration,
    }))
}

async fn request_preview_frame(
    address: SocketAddr,
    request: [u8; 8],
    settings: &FramePreviewStreamSettings,
) -> Option<IdentifiedMjpegFrame> {
    tokio::time::timeout(settings.request_timeout, async {
        let mut stream = TcpStream::connect(address).await.ok()?;
        stream.write_all(&request).await.ok()?;
        let mut length_bytes = [0_u8; 4];
        stream.read_exact(&mut length_bytes).await.ok()?;
        let header_length = u32::from_be_bytes(length_bytes) as usize;
        if header_length == 0 || header_length > settings.maximum_header_bytes {
            return None;
        }
        let mut header_bytes = vec![0_u8; header_length];
        stream.read_exact(&mut header_bytes).await.ok()?;
        let header: FramePreviewResponseHeaderV1 = serde_json::from_slice(&header_bytes).ok()?;
        if header.schema_version != 1 || header.status != FramePreviewStatusV1::Ok {
            return None;
        }
        let descriptor = header.descriptor?;
        let jpeg_length = usize::try_from(header.jpeg_bytes).ok()?;
        if jpeg_length < 4 || jpeg_length > settings.maximum_jpeg_bytes {
            return None;
        }
        let mut jpeg = vec![0_u8; jpeg_length];
        stream.read_exact(&mut jpeg).await.ok()?;
        if jpeg[..2] != [0xff, 0xd8] || jpeg[jpeg_length - 2..] != [0xff, 0xd9] {
            return None;
        }
        Some(IdentifiedMjpegFrame { descriptor, jpeg })
    })
    .await
    .ok()
    .flatten()
}

async fn health_live() -> Json<Value> {
    Json(json!({"status": "alive"}))
}

async fn health_ready(State(gateway): State<Gateway>) -> Response {
    match gateway.snapshot_now() {
        Some(snapshot)
            if snapshot.current.localization_received_monotonic_ns.is_some()
                && snapshot.current.freshness == StateFreshness::Fresh => Json(json!({
            "status": "ready",
            "localization_available": true,
            "freshness": snapshot.current.freshness,
        }))
        .into_response(),
        Some(snapshot) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "localization_available": snapshot.current.localization_received_monotonic_ns.is_some(),
                "freshness": snapshot.current.freshness,
            })),
        )
            .into_response(),
        None => not_ready_response(),
    }
}

async fn diagnostics_websocket_upgrade(
    State(gateway): State<Gateway>,
    upgrade: WebSocketUpgrade,
) -> Response {
    // Subscribe before upgrade completion; watch retains only the newest full
    // snapshot, including any restart/evidence transition during the handshake.
    let snapshots = gateway.inner.diagnostics_snapshots.subscribe();
    let permit = match Arc::clone(&gateway.inner.diagnostics_websocket_slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "diagnostics websocket capacity reached",
            )
                .into_response()
        }
    };
    upgrade
        .on_upgrade(move |socket| run_diagnostics_websocket(socket, snapshots, gateway, permit))
        .into_response()
}

async fn run_diagnostics_websocket(
    mut socket: WebSocket,
    mut snapshots: watch::Receiver<BrowserDiagnosticsSnapshot>,
    gateway: Gateway,
    _permit: tokio::sync::OwnedSemaphorePermit,
) {
    let send_deadline = gateway.inner.config.websocket_send_deadline;
    let initial = snapshots.borrow_and_update().clone();
    if !send_diagnostics_snapshot(&mut socket, &initial, send_deadline).await {
        return;
    }
    let mut last_revision = initial.gateway.revision;
    loop {
        tokio::select! {
            changed = snapshots.changed() => {
                if changed.is_err() {
                    return;
                }
                let next = snapshots.borrow_and_update().clone();
                if next.gateway.revision <= last_revision {
                    continue;
                }
                if !send_diagnostics_snapshot(&mut socket, &next, send_deadline).await {
                    return;
                }
                last_revision = next.gateway.revision;
            }
            message = socket.recv() => {
                match message {
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                    Some(Ok(Message::Ping(payload))) => {
                        let result = within_send_deadline(
                            send_deadline,
                            socket.send(Message::Pong(payload)),
                        ).await;
                        if !matches!(result, Some(Ok(()))) {
                            return;
                        }
                    }
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

async fn send_diagnostics_snapshot(
    socket: &mut WebSocket,
    snapshot: &BrowserDiagnosticsSnapshot,
    deadline: Duration,
) -> bool {
    let Ok(encoded) = serde_json::to_string(snapshot) else {
        return false;
    };
    matches!(
        within_send_deadline(deadline, socket.send(Message::Text(encoded.into()))).await,
        Some(Ok(()))
    )
}

fn not_ready_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "status": "not_ready",
            "error": "localization_unavailable",
            "message": "No localized scene is available yet.",
            "perception_source": "rust_scene",
        })),
    )
        .into_response()
}

async fn websocket_upgrade(
    State(gateway): State<Gateway>,
    Query(query): Query<WebSocketQuery>,
    upgrade: WebSocketUpgrade,
) -> Response {
    // Subscribe before validating/snapshotting so an update or producer restart between
    // the handshake and `on_upgrade` is represented by the watch channel's latest value.
    let latest = gateway.inner.latest.subscribe();
    if gateway.snapshot_now().is_none() {
        return not_ready_response();
    }
    let permit = match Arc::clone(&gateway.inner.websocket_slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"status":"unavailable","error":"websocket_capacity_reached"})),
            )
                .into_response()
        }
    };
    let max_message_size = gateway.inner.config.maximum_scene_bytes;
    let retention_duration = Duration::from_millis(query.retain_last_known_ms.unwrap_or_default());
    let maximum_balls = StateLifecycleConfig::default().maximum_balls;
    upgrade
        .max_message_size(max_message_size)
        .max_frame_size(max_message_size)
        .on_upgrade(move |socket| {
            websocket_session(
                socket,
                gateway,
                latest,
                permit,
                retention_duration,
                maximum_balls,
            )
        })
        .into_response()
}

async fn websocket_session(
    mut socket: WebSocket,
    gateway: Gateway,
    mut latest: watch::Receiver<Option<Arc<PublishedState>>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
    retention_duration: Duration,
    maximum_balls: usize,
) {
    let mut position_retention = WebSocketPositionRetention::new(retention_duration, maximum_balls);
    // Allocate per-client counters only under the explicitly enabled gateway audit.
    let mut audit_client =
        gateway
            .inner
            .gateway_audit
            .as_ref()
            .map(|audit| GatewayWebSocketAuditClient {
                audit: Arc::clone(audit),
                client_id: audit.next_client_id.fetch_add(1, Ordering::Relaxed),
                previous_publication_sequence: None,
                previous_source_identity: None,
            });
    // `borrow_and_update` selects the newest state even if several updates landed while
    // the WebSocket handshake was completing. Intermediate scenes are intentionally lost.
    let Some(initial) = take_latest_state(&mut latest) else {
        let _ = within_send_deadline(
            gateway.inner.config.websocket_send_deadline,
            socket.send(Message::Close(Some(CloseFrame {
                code: 1012,
                reason: "scene producer restarted".into(),
            }))),
        )
        .await;
        return;
    };
    if !send_websocket_state(
        &mut socket,
        &initial,
        &gateway,
        audit_client.as_mut(),
        None,
        gateway.inner.config.websocket_send_deadline,
        &mut position_retention,
    )
    .await
    {
        return;
    }
    loop {
        let expiration_deadline = position_retention.expiration_deadline();
        let retention_expiration = async move {
            if let Some(deadline) = expiration_deadline {
                tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        tokio::pin!(retention_expiration);
        tokio::select! {
            changed = latest.changed() => {
                let wake_started = Instant::now();
                if changed.is_err() {
                    return;
                }
                let next = take_latest_state(&mut latest);
                let wakeup_to_consume_ms = wake_started.elapsed().as_secs_f64() * 1_000.0;
                match next {
                    Some(state) => {
                        if !send_websocket_state(
                            &mut socket,
                            &state,
                            &gateway,
                            audit_client.as_mut(),
                            Some(wakeup_to_consume_ms),
                            gateway.inner.config.websocket_send_deadline,
                            &mut position_retention,
                        ).await {
                            return;
                        }
                    }
                    None => {
                        let _ = within_send_deadline(
                            gateway.inner.config.websocket_send_deadline,
                            socket.send(Message::Close(Some(CloseFrame {
                                code: 1012,
                                reason: "scene producer restarted".into(),
                            }))),
                        )
                        .await;
                        return;
                    }
                }
            }
            _ = &mut retention_expiration => {
                // Publish an explicit expired state even if the producer goes quiet;
                // otherwise consumers could keep displaying the layout beyond grace.
                let Some(state) = gateway.snapshot_now() else {
                    continue;
                };
                if !send_websocket_state(
                    &mut socket,
                    &state,
                    &gateway,
                    audit_client.as_mut(),
                    None,
                    gateway.inner.config.websocket_send_deadline,
                    &mut position_retention,
                ).await {
                    return;
                }
            }
            message = socket.recv() => {
                match message {
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                    Some(Ok(Message::Ping(payload))) => {
                        let pong = within_send_deadline(
                            gateway.inner.config.websocket_send_deadline,
                            socket.send(Message::Pong(payload)),
                        ).await;
                        if !matches!(pong, Some(Ok(()))) {
                            return;
                        }
                    }
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}

fn take_latest_state(
    receiver: &mut watch::Receiver<Option<Arc<PublishedState>>>,
) -> Option<Arc<PublishedState>> {
    receiver.borrow_and_update().clone()
}

async fn send_websocket_state(
    socket: &mut WebSocket,
    state: &PublishedState,
    gateway: &Gateway,
    mut audit_client: Option<&mut GatewayWebSocketAuditClient>,
    watch_wakeup_to_consume_ms: Option<f64>,
    send_deadline: Duration,
    position_retention: &mut WebSocketPositionRetention,
) -> bool {
    // Capture selection age before assembly/serialization/write so those costs
    // remain disjoint stages instead of leaking into publication-to-selection.
    let publication_to_selection_ms = gateway
        .monotonic_ns()
        .saturating_sub(state.published_at_monotonic_ns)
        as f64
        / 1_000_000.0;
    let assembly_started = Instant::now();
    let message = websocket_scene_message_with_retention(state, position_retention);
    let message_assembly_ms = assembly_started.elapsed().as_secs_f64() * 1_000.0;
    let serialization_started = Instant::now();
    let encoded = match serde_json::to_string(&message) {
        Ok(encoded) => encoded,
        Err(_) => {
            if let Some(audit_client) = audit_client.as_deref_mut() {
                audit_client.record_send(
                    state,
                    publication_to_selection_ms,
                    watch_wakeup_to_consume_ms,
                    message_assembly_ms,
                    serialization_started.elapsed().as_secs_f64() * 1_000.0,
                    None,
                    None,
                    "serialization_failed",
                );
            }
            return false;
        }
    };
    let json_serialization_ms = serialization_started.elapsed().as_secs_f64() * 1_000.0;
    let payload_bytes = encoded.len();
    let send_started = Instant::now();
    let send_result =
        within_send_deadline(send_deadline, socket.send(Message::Text(encoded.into()))).await;
    let socket_send_call_ms = send_started.elapsed().as_secs_f64() * 1_000.0;
    let outcome = match send_result {
        Some(Ok(())) => "sent",
        Some(Err(_)) => "send_failed",
        None => "deadline_exceeded",
    };
    if let Some(audit_client) = audit_client {
        audit_client.record_send(
            state,
            publication_to_selection_ms,
            watch_wakeup_to_consume_ms,
            message_assembly_ms,
            json_serialization_ms,
            Some(socket_send_call_ms),
            Some(payload_bytes),
            outcome,
        );
    }
    outcome == "sent"
}

impl GatewayWebSocketAuditClient {
    #[allow(clippy::too_many_arguments)]
    fn record_send(
        &mut self,
        state: &PublishedState,
        publication_to_selection_ms: f64,
        watch_wakeup_to_consume_ms: Option<f64>,
        message_assembly_ms: f64,
        json_serialization_ms: f64,
        socket_send_call_ms: Option<f64>,
        payload_bytes: Option<usize>,
        outcome: &'static str,
    ) {
        let status = &state.current.payload.observation_status;
        let engine_instance_id = status.get("engine_instance_id").and_then(Value::as_str);
        let source_session_id = status.get("source_session_id").and_then(Value::as_str);
        let source_frame_id = status.get("source_frame_id").and_then(Value::as_str);
        let transport_sequence = status.get("transport_sequence").and_then(Value::as_u64);
        let source_identity = match (engine_instance_id, source_session_id, transport_sequence) {
            (Some(engine), Some(session), Some(sequence)) => {
                Some((engine.to_owned(), session.to_owned(), sequence))
            }
            _ => None,
        };
        let repeated_source_observation = source_identity
            .as_ref()
            .zip(self.previous_source_identity.as_ref())
            .is_some_and(|(current, previous)| {
                current.0 == previous.0 && current.1 == previous.1 && current.2 == previous.2
            });
        let source_sequence_gap = source_identity
            .as_ref()
            .zip(self.previous_source_identity.as_ref())
            .and_then(|(current, previous)| {
                (current.0 == previous.0 && current.1 == previous.1)
                    .then(|| current.2.saturating_sub(previous.2).saturating_sub(1))
                    .filter(|gap| *gap > 0)
            });
        if source_identity.is_some() {
            self.previous_source_identity = source_identity;
        }

        let repeated_publication =
            self.previous_publication_sequence == Some(state.publication_sequence);
        let publication_sequence_gap = self
            .previous_publication_sequence
            .map(|previous| {
                state
                    .publication_sequence
                    .saturating_sub(previous)
                    .saturating_sub(1)
            })
            .unwrap_or_default();
        self.previous_publication_sequence = Some(state.publication_sequence);
        self.audit
            .recorder
            .record_gateway_websocket_send(LiveAuditGatewayWebSocketSend {
                run_id: &self.audit.run_id,
                gateway_instance_id: &self.audit.gateway_instance_id,
                monotonic_clock_domain_id: &self.audit.monotonic_clock_domain_id,
                client_id: self.client_id,
                publication_sequence: state.publication_sequence,
                publication_to_selection_ms,
                watch_wakeup_to_consume_ms,
                message_assembly_ms,
                json_serialization_ms,
                socket_send_call_ms,
                payload_bytes,
                outcome,
                repeated_publication,
                repeated_source_observation,
                publication_sequence_gap,
                source_sequence_gap,
                engine_instance_id,
                source_session_id,
                source_frame_id,
                transport_sequence,
                snapshot_revision: status.get("snapshot_revision").and_then(Value::as_u64),
                configuration_revision: status
                    .get("configuration_revision")
                    .and_then(Value::as_u64),
                rectification_revision: status
                    .get("rectification_revision")
                    .and_then(Value::as_u64),
                calibration_revision: status.get("calibration_revision").and_then(Value::as_u64),
                freshness: status.get("freshness_status").and_then(Value::as_str),
                ball_positions_status: status.get("ball_positions_status").and_then(Value::as_str),
            });
    }
}

async fn within_send_deadline<F: Future>(deadline: Duration, future: F) -> Option<F::Output> {
    timeout(deadline, future).await.ok()
}

fn websocket_scene_message(state: &PublishedState) -> Value {
    let timestamp = timestamp_text(state.published_at);
    let sequence = state.publication_sequence;
    let transport_sequence = transport_sequence(&state.current.payload.observation_status);
    let balls = state
        .current
        .payload
        .balls
        .iter()
        .map(|ball| {
            let mut value = serde_json::to_value(ball)?;
            if let Some(object) = value.as_object_mut() {
                object.entry("table_position").or_insert(Value::Null);
            }
            Ok::<_, serde_json::Error>(value)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_default();
    json!({
        "type": "state",
        "timestamp": timestamp,
        "sequence": sequence,
        "data": {
            "ball_count": balls.len(),
            "balls": balls,
            "cue": null,
            "table": state.current.payload.table,
            "game_type": state.current.payload.game_type,
            "frame_number": transport_sequence,
            "observation_status": state.current.payload.observation_status,
            "sequence": sequence,
            "timestamp": timestamp,
        }
    })
}

fn websocket_scene_message_with_retention(
    state: &PublishedState,
    retention: &mut WebSocketPositionRetention,
) -> Value {
    let mut message = websocket_scene_message(state);
    if retention.duration.is_zero() {
        return message;
    }

    let status = &state.current.payload.observation_status;
    // This compatibility query option is retained for old clients, but the
    // phase-aware producer owns its complete trusted/stale ball layout.
    if status
        .pointer("/perception/phase")
        .and_then(Value::as_str)
        .is_some()
    {
        message["data"]["observation_status"]["position_retention"] = Value::Null;
        return message;
    }
    let current_context = position_retention_context(status);
    let current_status = status
        .get("ball_positions_status")
        .cloned()
        .unwrap_or(Value::Null);

    // Context fields may be absent in a producer-health state. A supplied field
    // must match the cached context; missing health-state fields alone do not
    // discard useful last-known positions.
    let context_changed = retention.layout.as_ref().is_some_and(|layout| {
        POSITION_RETENTION_CONTEXT_FIELDS.iter().any(|field| {
            status
                .get(*field)
                .is_some_and(|value| retention_context_field(&layout.context, field) != Some(value))
        })
    });
    if context_changed || empty_layout_is_authoritative(status) {
        retention.layout = None;
        retention.unavailable_since = None;
    }

    let fresh_observed_layout = current_state_is_observed_layout(state);
    if fresh_observed_layout && retention_context_has_source(&current_context) {
        let mut balls = state.current.payload.balls.clone();
        balls.truncate(retention.maximum_balls);
        let source_identity = position_retention_source_identity(status);
        let same_source_observation = retention
            .layout
            .as_ref()
            .is_some_and(|layout| layout.source_identity == source_identity);
        if !same_source_observation {
            retention.layout = Some(RetainedBallLayout {
                balls,
                captured_at: Instant::now(),
                context: current_context,
                original_status: current_status.clone(),
                source_identity,
            });
        }
        retention.unavailable_since = None;
    } else if fresh_observed_layout {
        // Never carry a known layout forward when a supposedly fresh observation
        // lacks enough producer identity to prove it belongs to the same source.
        retention.layout = None;
        retention.unavailable_since = None;
    } else if retention.layout.is_some() {
        // Start one grace interval at the first unavailable state. Repeated
        // failures do not extend it, and time spent in a healthy observed state
        // does not consume the interval.
        retention.unavailable_since.get_or_insert_with(Instant::now);
    }

    let mut retained_age_ms = None;
    let mut unavailable_age_ms = None;
    let mut original_status = Value::Null;
    let mut source_identity = Value::Null;
    let mut retained = false;
    if let Some(layout) = retention.layout.as_ref() {
        let layout_age = layout.captured_at.elapsed();
        retained_age_ms = Some(duration_millis_saturated(layout_age));
        let unavailable_age = retention.unavailable_since.map(|start| start.elapsed());
        unavailable_age_ms = unavailable_age.map(duration_millis_saturated);
        original_status = layout.original_status.clone();
        source_identity = layout.source_identity.clone();
        if unavailable_age.is_some_and(|age| age >= retention.duration) {
            retention.layout = None;
            retention.unavailable_since = None;
        } else if !fresh_observed_layout && unavailable_age.is_some() {
            if let Some(layout) = retention.layout.as_ref() {
                let balls = retained_layout_values(layout, retention.maximum_balls);
                message["data"]["ball_count"] = json!(balls.len());
                message["data"]["balls"] = json!(balls);
                message["data"]["observation_status"]["ball_positions_status"] = json!("cached");
                retained = true;
            }
        }
    }

    message["data"]["observation_status"]["position_retention"] = json!({
        "retained": retained,
        "retained_age_ms": retained_age_ms,
        "unavailable_age_ms": unavailable_age_ms,
        "grace_ms": duration_millis_saturated(retention.duration),
        "current_status": current_status,
        "original_status": original_status,
        "source_identity": source_identity,
    });
    message
}

fn current_state_is_observed_layout(state: &PublishedState) -> bool {
    let status = &state.current.payload.observation_status;
    state.current.freshness == StateFreshness::Fresh
        && status.get("freshness_status").and_then(Value::as_str) == Some("fresh")
        && status.get("ball_positions_status").and_then(Value::as_str) == Some("observed")
        && status.get("ball_detection_status").and_then(Value::as_str) == Some("observed")
        && !state.current.payload.balls.is_empty()
        && state
            .current
            .payload
            .balls
            .iter()
            .all(|ball| ball.is_observed)
}

fn empty_layout_is_authoritative(status: &Value) -> bool {
    status
        .get("perception")
        .and_then(|perception| perception.get("empty_layout_verified"))
        .and_then(Value::as_bool)
        == Some(true)
}

fn position_retention_context(status: &Value) -> PositionRetentionContext {
    let field = |name| status.get(name).cloned();
    PositionRetentionContext {
        engine_instance_id: field("engine_instance_id"),
        source_session_id: field("source_session_id"),
        configuration_revision: field("configuration_revision"),
        rectification_revision: field("rectification_revision"),
        calibration_revision: field("calibration_revision"),
        calibration_state: field("calibration_state"),
    }
}

fn retention_context_field<'a>(
    context: &'a PositionRetentionContext,
    field: &str,
) -> Option<&'a Value> {
    match field {
        "engine_instance_id" => context.engine_instance_id.as_ref(),
        "source_session_id" => context.source_session_id.as_ref(),
        "configuration_revision" => context.configuration_revision.as_ref(),
        "rectification_revision" => context.rectification_revision.as_ref(),
        "calibration_revision" => context.calibration_revision.as_ref(),
        "calibration_state" => context.calibration_state.as_ref(),
        _ => None,
    }
}

fn retention_context_has_source(context: &PositionRetentionContext) -> bool {
    context
        .engine_instance_id
        .as_ref()
        .and_then(Value::as_str)
        .is_some_and(|engine| !engine.is_empty())
        && context
            .source_session_id
            .as_ref()
            .and_then(Value::as_str)
            .is_some_and(|session| !session.is_empty())
}

fn position_retention_source_identity(status: &Value) -> Value {
    let mut identity = serde_json::Map::new();
    for field in [
        "engine_instance_id",
        "source_session_id",
        "source_frame_id",
        "transport_sequence",
        "snapshot_revision",
        "configuration_revision",
        "rectification_revision",
        "calibration_revision",
        "calibration_state",
    ] {
        if let Some(value) = status.get(field) {
            identity.insert(field.to_owned(), value.clone());
        }
    }
    Value::Object(identity)
}

fn retained_layout_values(layout: &RetainedBallLayout, maximum_balls: usize) -> Vec<Value> {
    layout
        .balls
        .iter()
        .take(maximum_balls)
        .filter_map(|ball| serde_json::to_value(ball).ok())
        .map(|mut ball| {
            if let Some(fields) = ball.as_object_mut() {
                fields.insert("is_observed".to_owned(), json!(false));
                fields.insert("tracking_status".to_owned(), json!("lost"));
                fields.insert("position_kind".to_owned(), json!("last_observed"));
                fields.insert("confidence".to_owned(), json!(0.0));
                fields.insert("confidence_kind".to_owned(), json!("unavailable"));
                fields.entry("table_position").or_insert(Value::Null);
            }
            ball
        })
        .collect()
}

fn duration_millis_saturated(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn http_scene_response(state: &PublishedState) -> Value {
    let status = &state.current.payload.observation_status;
    let confidence_values: Vec<f64> = state
        .current
        .payload
        .balls
        .iter()
        .filter(|ball| ball.confidence_kind == "detector_observation")
        .map(|ball| ball.confidence)
        .collect();
    let confidence = if confidence_values.is_empty() {
        None
    } else {
        Some(confidence_values.iter().sum::<f64>() / confidence_values.len() as f64)
    };
    let balls = state
        .current
        .payload
        .balls
        .iter()
        .map(|ball| {
            let mut value = serde_json::to_value(ball).unwrap_or_else(|_| json!({}));
            if let Some(object) = value.as_object_mut() {
                let ball_type = object.remove("type").unwrap_or(Value::Null);
                object.insert("ball_type".into(), ball_type);
                object.entry("table_position").or_insert(Value::Null);
                object.insert("velocity".into(), Value::Null);
                object.insert("is_cue_ball".into(), Value::Null);
                object.insert("is_pocketed".into(), Value::Null);
                let age_ns = state
                    .current
                    .ball_age_ns
                    .get(&ball.id)
                    .copied()
                    .unwrap_or_default();
                object.insert(
                    "last_update".into(),
                    Value::String(timestamp_text(
                        state.snapshot_at_wall - chrono_duration_from_ns(age_ns),
                    )),
                );
                object.insert("velocity_status".into(), json!("unavailable"));
                object.insert("cue_ball_status".into(), json!("unavailable"));
                object.insert("pocket_status".into(), json!("unavailable"));
            }
            value
        })
        .collect::<Vec<_>>();
    let frame_number = transport_sequence(status);
    // Preserve the public state contract's established validity predicate.
    // Freshness is independently visible in observation_status; it is not
    // part of this contract field.
    let is_valid = scene_state_is_valid(status);
    json!({
        "timestamp": timestamp_text(state.published_at),
        "frame_number": frame_number,
        "balls": balls,
        "cue": null,
        "table": status.pointer("/perception/table_projection").cloned().unwrap_or(Value::Null),
        "game_type": null,
        "session": null,
        "is_valid": is_valid,
        "confidence": confidence,
        "core_metadata_status": "unavailable",
        "core_metadata_timestamp": null,
        "table_metadata_status": "unavailable",
        "scene_calibration_revision": status.get("calibration_revision").cloned().unwrap_or(Value::Null),
        "table_calibration_revision": status.pointer("/perception/table_projection/calibration_revision").cloned().unwrap_or(Value::Null),
        "events": [],
        "confidence_status": if confidence.is_some() { "mean_detector_confidence" } else { "unavailable" },
        "observation_status": status,
        "perception_source": "rust_scene",
    })
}

fn scene_state_is_valid(status: &Value) -> bool {
    // Stale and cached status remains visible in observation_status and also
    // makes the position check fail.
    status.get("phase").and_then(Value::as_str) == Some("rust_scene_observation")
        && status.get("ball_detection_status").and_then(Value::as_str) == Some("observed")
        && status.get("ball_positions_status").and_then(Value::as_str) == Some("observed")
}

fn transport_sequence(status: &Value) -> Option<u64> {
    status.get("transport_sequence").and_then(Value::as_u64)
}

fn timestamp_text(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn chrono_duration_from_ns(value: u128) -> chrono::Duration {
    let nanoseconds = i64::try_from(value).unwrap_or(i64::MAX);
    chrono::Duration::nanoseconds(nanoseconds)
}

async fn run_freshness_scheduler(gateway: Gateway) {
    let mut latest = gateway.inner.latest.subscribe();
    loop {
        let published = latest.borrow_and_update().clone();
        let Some(published) = published else {
            if latest.changed().await.is_err() {
                return;
            }
            continue;
        };
        let delay = gateway.freshness_delay(&published);
        match delay {
            Some(delay) => tokio::select! {
                _ = sleep(delay) => gateway.refresh_freshness(),
                changed = latest.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
            },
            None => {
                if latest.changed().await.is_err() {
                    return;
                }
            }
        }
    }
}

/// Publish complete newest-only snapshots. Evidence changes wake this task;
/// the configured timer handles browser heartbeats and exact producer expiry or
/// active-work deadlines without polling any HTTP or scene endpoint.
async fn run_diagnostics_snapshot_publisher(gateway: Gateway) {
    let mut reports = gateway.inner.diagnostics.subscribe();
    let mut scenes = gateway.inner.latest.subscribe();
    let mut connections = gateway.inner.producer_connection_updates.subscribe();
    let mut shutdown = gateway.inner.shutdown.subscribe();
    loop {
        gateway.publish_diagnostics_snapshot();
        let delay = gateway.diagnostics_next_delay();
        tokio::select! {
            changed = reports.changed() => {
                if changed.is_err() {
                    return;
                }
                let _ = reports.borrow_and_update();
            }
            changed = scenes.changed() => {
                if changed.is_err() {
                    return;
                }
                let _ = scenes.borrow_and_update();
            }
            changed = connections.changed() => {
                if changed.is_err() {
                    return;
                }
                let _ = connections.borrow_and_update();
            }
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow_and_update() {
                    return;
                }
            }
            _ = sleep(delay) => {}
        }
    }
}

/// Start HTTP and live-v3 loopback scene listeners until shutdown.
pub async fn run(config: ServerConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    run_with_live(config, LiveProcessLaunchOptions::default()).await
}

fn configured_setup_source(configuration: &Value) -> Option<&str> {
    configuration
        .pointer("/perception/capture/device_selector")
        .and_then(Value::as_str)
        .filter(|source| crate::configuration_authority::is_gstreamer_selector(source))
        .or_else(|| {
            configuration
                .pointer("/vision/camera/device_id")
                .and_then(Value::as_str)
                .filter(|source| crate::configuration_authority::valid_http_uri(source))
        })
}

fn setup_failure(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error": message.to_string()}))).into_response()
}

fn setup_table_calibrated(status: &Value) -> bool {
    // Scene calibration describes precise ball coordinates, which require
    // separate optics and ball-center assets. A visibility comparison is
    // evidence that the producer loaded the operator's table reference.
    status["calibration_state"] == "valid"
        || matches!(
            status
                .pointer("/perception/table_visibility/reason")
                .and_then(Value::as_str),
            Some(
                "reference_match"
                    | "extreme_brightness_shift"
                    | "extreme_color_shift"
                    | "lost_rail_contrast"
                    | "cloth_mismatch"
                    | "rail_mismatch"
            )
        )
}

fn precision_capture(
    configuration: &Value,
    root: &std::path::Path,
) -> anyhow::Result<crate::capture_settings_wire::ResolvedCaptureSettings> {
    use crate::capture_settings_wire::{build_capture_settings_v3, parse_capture_settings_json};
    let live = &configuration["perception"]["rust_live"];
    let request = build_capture_settings_v3(configuration)?;
    let mut capture = parse_capture_settings_json(&request.to_string())?;
    if live["frame_preview_enabled"] == true
        && live["frame_preview_calibration_control_enabled"] == true
    {
        if let (Some(path), Some(timeout), Some(maximum)) = (
            live["frame_preview_calibration_store_path"].as_str(),
            live["frame_preview_calibration_store_lock_timeout_ms"].as_u64(),
            live["frame_preview_calibration_store_max_file_bytes"].as_u64(),
        ) {
            let store = crate::calibration_store::CalibrationStore::new(
                root.join(path),
                Duration::from_millis(timeout),
                usize::try_from(maximum)?,
            )?;
            crate::live_application::install_calibration_store_state(&mut capture, &store);
        }
    }
    Ok(capture)
}

/// Evaluate configured files against the same capture and table context used
/// by the producer. This is independent of whether any balls are visible.
fn setup_precision_calibration(configuration: &Value, root: &std::path::Path) -> Value {
    use crate::ball_localization_profile::{
        BallLocalizationSettings, LoadedBallLocalizationProfile,
    };
    let live = &configuration["perception"]["rust_live"];
    let mut settings: BallLocalizationSettings =
        match serde_json::from_value(live["ball_localization"].clone()) {
            Ok(settings) => settings,
            Err(_) => {
                return json!({"optics": {"state": "invalid"}, "ball_centers": {"state": "invalid"}})
            }
        };
    if let Ok(request) = crate::capture_settings_wire::build_capture_settings_v3(configuration) {
        if let Some(profile) = request
            .pointer("/requested/camera_profile_id")
            .and_then(Value::as_str)
        {
            crate::browser_precision_calibration::select_assets(&mut settings, root, profile);
        }
    }
    let mut optics = json!({"state": if settings.optics_profile_path.is_some() { "unverified" } else { "missing" }});
    let mut centers = json!({"state": if settings.ball_center_calibration_path.is_some() { "unverified" } else { "missing" }});
    if settings.optics_profile_path.is_some() {
        let check = || -> anyhow::Result<_> {
            let capture = precision_capture(configuration, root)?;
            let expected = &capture.expected;
            let (width, height) = expected
                .output_frame_geometry()
                .map(|geometry| (geometry.width_px(), geometry.height_px()))
                .unwrap_or_else(|| {
                    let profile = expected.expected_negotiated_profile();
                    (profile.width_px(), profile.height_px())
                });
            let mut settings = settings.clone();
            settings.resolve_paths(root);
            let profile = LoadedBallLocalizationProfile::load_with_table(
                settings,
                expected.camera_profile_id(),
                expected.image_space(),
                width,
                height,
                expected.rectification_revision(),
                expected.calibration(),
            );
            Ok((profile, expected.calibration().is_some()))
        };
        match check() {
            Ok((Ok(profile), _)) if profile.optics.is_none() => {
                // Loading a usable coarse/table-only profile is not evidence
                // that the optional precision asset itself is ready.
                optics = json!({"state": "invalid", "detail": profile.calibration_error});
                if settings.ball_center_calibration_path.is_some() {
                    centers = json!({"state": "blocked", "detail": "Fix the camera optics profile first."});
                }
            }
            Ok((Ok(profile), table_active)) => {
                optics = json!({"state": "ready"});
                if settings.ball_center_calibration_path.is_some() {
                    centers = if !table_active {
                        json!({"state": "blocked", "detail": "Calibrate the table for the current camera before validating ball centers."})
                    } else if let Some(error) = profile.calibration_error {
                        json!({"state": "invalid", "detail": error})
                    } else if profile.calibration.is_some() {
                        json!({"state": "ready"})
                    } else {
                        centers
                    };
                }
            }
            Ok((Err(error), _)) => {
                optics = json!({"state": "invalid", "detail": error.to_string()});
                if settings.ball_center_calibration_path.is_some() {
                    centers = json!({"state": "blocked", "detail": "Fix the camera optics profile first."});
                }
            }
            Err(error) => {
                optics["detail"] = json!(error.to_string());
            }
        }
    }
    json!({"optics": optics, "ball_centers": centers})
}

async fn setup_status(State(gateway): State<Gateway>) -> Response {
    let Ok(_permit) = gateway
        .inner
        .configuration_slots
        .clone()
        .try_acquire_owned()
    else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Camera setup is busy");
    };
    let Some(authority) = gateway.inner.configuration_authority.clone() else {
        return Json(json!({"state": "disabled"})).into_response();
    };
    let root = gateway.inner.config.project_root.clone();
    let (configuration, precision_calibration) = match tokio::task::spawn_blocking(move || {
        let configuration = authority.effective_configuration()?;
        let root = root
            .map_or_else(std::env::current_dir, Ok)
            .map_err(|error| error.to_string())?;
        let precision = setup_precision_calibration(&configuration, &root);
        Ok::<_, String>((configuration, precision))
    })
    .await
    {
        Ok(Ok(result)) => result,
        _ => {
            return setup_failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "Unable to read setup configuration",
            )
        }
    };
    let source = configured_setup_source(&configuration);
    let diagnostics = gateway.inner.diagnostics.snapshot();
    let stage_error = diagnostics
        .report
        .as_ref()
        .or_else(|| {
            diagnostics
                .last_known
                .as_ref()
                .filter(|_| diagnostics.exited)
                .map(|last| &last.report)
        })
        .and_then(|report| {
            report.stages.iter().find(|stage| {
                stage.status == crate::producer_diagnostics_contract::StageStatus::Failed
            })
        })
        .map(|stage| {
            stage
                .detail
                .clone()
                .unwrap_or_else(|| stage.reason_code.clone())
        });
    let snapshot = gateway.snapshot_now();
    let revision = configuration["metadata"]["configuration_revision"].as_u64();
    // Cue and visibility evidence can arrive before the first localization.
    // The validated producer handshake identifies the active revision in both cases.
    let producer_revision = lock_unpoisoned(&gateway.inner.state).live_configuration_revision;
    let connected = gateway.inner.producer_connected.load(Ordering::Acquire)
        && revision.is_some()
        && producer_revision == revision
        && snapshot.is_some();
    let error = stage_error.or_else(|| {
        if connected {
            None
        } else {
            lock_unpoisoned(&gateway.inner.setup_error).clone()
        }
    });
    let state = if !gateway.inner.setup_enabled.load(Ordering::Acquire) {
        "disabled"
    } else if source.is_none() {
        "idle"
    } else if connected {
        "running"
    } else if error.is_some() {
        "error"
    } else {
        "starting"
    };
    (
        [(CACHE_CONTROL, "no-store")],
        Json(json!({
            "state": state,
            "error": error,
            "configured_source": source,
            "precision_calibration": precision_calibration,
            "configuration_revision": configuration.pointer("/metadata/configuration_revision"),
            "calibrated": connected && snapshot.as_ref().is_some_and(|snapshot|
                setup_table_calibrated(&snapshot.current.payload.observation_status)),
        })),
    )
        .into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetupSource {
    expected_revision: u64,
    source: String,
}

async fn setup_source(
    State(gateway): State<Gateway>,
    Json(request): Json<SetupSource>,
) -> Response {
    if !gateway.inner.setup_enabled.load(Ordering::Acquire) {
        return setup_failure(StatusCode::SERVICE_UNAVAILABLE, "Live capture is disabled");
    }
    let source = request.source.trim();
    let physical = crate::configuration_authority::is_gstreamer_selector(source);
    if !physical && !crate::configuration_authority::valid_http_uri(source) {
        return setup_failure(
            StatusCode::BAD_REQUEST,
            "Choose a discovered camera or enter an HTTP(S) MJPEG URL",
        );
    }
    let Ok(permit) = gateway
        .inner
        .configuration_slots
        .clone()
        .try_acquire_owned()
    else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Camera setup is busy");
    };
    let Some(authority) = gateway.inner.configuration_authority.clone() else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Setup storage is unavailable",
        );
    };
    let expected_revision = request.expected_revision;
    let patch = json!({
        "perception": {"capture": {"device_selector": physical.then_some(source)}},
        "vision": {"camera": {"device_id": source, "video_file_path": null}}
    });
    // Once accepted, finish the store write and notify the supervisor even
    // if the browser disconnects before reading the response.
    let response = tokio::spawn(async move {
        let _permit = permit;
        let _guard = gateway.inner.configuration_mutation_lock.lock().await;
        let response = tokio::task::spawn_blocking(move || {
            authority.update_setup_source(expected_revision, patch)
        })
        .await;
        if let Ok(response) = &response {
            if response["ok"] == true {
                *lock_unpoisoned(&gateway.inner.setup_error) = None;
                gateway.inner.setup_reload.send_replace(());
            }
        }
        response
    })
    .await;
    let Ok(Ok(response)) = response else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Unable to save the camera source",
        );
    };
    if response["ok"] != true {
        let error = &response["error"];
        let status = match error["code"].as_str() {
            Some("revision_conflict") => StatusCode::CONFLICT,
            Some("store_error" | "lock_timeout") => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::BAD_REQUEST,
        };
        return setup_failure(
            status,
            error["message"]
                .as_str()
                .unwrap_or("Unable to save the camera source"),
        );
    }
    Json(json!({
        "state": "starting", "configured_source": source,
        "configuration_revision": response["revision"], "calibrated": false,
    }))
    .into_response()
}

async fn setup_calibration_frame(State(gateway): State<Gateway>) -> Response {
    if !gateway.inner.setup_enabled.load(Ordering::Acquire) {
        return setup_failure(StatusCode::SERVICE_UNAVAILABLE, "Live capture is disabled");
    }
    let Some(settings) = gateway.inner.frame_preview_stream.as_ref() else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Calibration preview is disabled",
        );
    };
    let Some(limits) = settings.calibration.as_ref() else {
        return setup_failure(StatusCode::SERVICE_UNAVAILABLE, "Calibration is disabled");
    };
    let Ok(_permit) = gateway
        .inner
        .configuration_slots
        .clone()
        .try_acquire_owned()
    else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Calibration is busy");
    };
    // Give calibration priority over the continuous preview so a stream
    // cannot occupy the producer's only encoder during the operator action.
    let _preview = gateway.inner.frame_preview_requests.write().await;
    let Ok(store) = crate::calibration_control::read_gateway_snapshot(
        settings.address,
        settings.request_timeout,
        limits.maximum_response_bytes,
    )
    .await
    else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Calibration store is unavailable",
        );
    };
    if store["status"] != "ok" {
        return calibration_control_failure(store);
    }
    let Some(frame) = request_preview_frame(settings.address, *PREVIEW_REQUEST_V1, settings).await
    else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "No calibration frame is available",
        );
    };
    let capture_token = uuid::Uuid::new_v4().to_string();
    {
        const RETAINED_CALIBRATION_STILLS: usize = 64;
        let mut stills = lock_unpoisoned(&gateway.inner.calibration_stills);
        stills.push_back((
            capture_token.clone(),
            crate::browser_precision_calibration::fingerprint(&frame.descriptor, &frame.jpeg),
        ));
        if stills.len() > RETAINED_CALIBRATION_STILLS {
            stills.pop_front();
        }
    }
    (
        [(CACHE_CONTROL, "no-store")],
        Json(json!({
            "descriptor": frame.descriptor,
            "capture_token": capture_token,
            "jpeg_base64": base64::engine::general_purpose::STANDARD.encode(frame.jpeg),
            "expected_store_revision": store["snapshot"]["revision"],
        })),
    )
        .into_response()
}

async fn setup_checkerboard_pdf() -> impl IntoResponse {
    const PDF: &[u8] =
        include_bytes!("../../output/pdf/ball-localization-checkerboard-letter-tiles.pdf");
    (
        [
            (axum::http::header::CONTENT_TYPE, "application/pdf"),
            (
                axum::http::header::CONTENT_DISPOSITION,
                "attachment; filename=ball-localization-checkerboard-letter-tiles.pdf",
            ),
        ],
        PDF,
    )
}

async fn setup_fit_optics(
    State(gateway): State<Gateway>,
    Json(request): Json<crate::browser_precision_calibration::OpticsRequest>,
) -> Response {
    setup_precision_operation(gateway, true, move |context| context.fit_optics(request)).await
}

async fn setup_fit_ball_centers(
    State(gateway): State<Gateway>,
    Json(request): Json<crate::browser_precision_calibration::BallRequest>,
) -> Response {
    setup_precision_operation(gateway, true, move |context| {
        context.fit_ball_centers(request)
    })
    .await
}

async fn setup_list_captures(
    State(gateway): State<Gateway>,
    Path(mode): Path<CaptureMode>,
) -> Response {
    setup_precision_operation(gateway, false, move |context| context.archive(mode)?.list()).await
}

async fn setup_store_capture(
    State(gateway): State<Gateway>,
    Path((mode, slot)): Path<(CaptureMode, usize)>,
    Json(capture): Json<Value>,
) -> Response {
    setup_precision_operation(gateway, false, move |context| {
        context.archive_capture(mode, slot, capture)
    })
    .await
}

async fn setup_remove_capture(
    State(gateway): State<Gateway>,
    Path((mode, slot)): Path<(CaptureMode, usize)>,
) -> Response {
    setup_precision_operation(gateway, false, move |context| {
        context.archive(mode)?.remove(Some(slot))
    })
    .await
}

async fn setup_clear_captures(
    State(gateway): State<Gateway>,
    Path(mode): Path<CaptureMode>,
) -> Response {
    setup_precision_operation(gateway, false, move |context| {
        context.archive(mode)?.remove(None)
    })
    .await
}

async fn setup_ball_candidate(
    State(gateway): State<Gateway>,
    Json(request): Json<crate::browser_precision_calibration::BallCandidateRequest>,
) -> Response {
    setup_precision_operation(gateway, false, move |context| {
        context.ball_candidate(request)
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BallPlacementQuery {
    x_mm: f64,
    y_mm: f64,
}

async fn setup_ball_placement(
    State(gateway): State<Gateway>,
    Query(query): Query<BallPlacementQuery>,
) -> Response {
    // Bound this optional preview independently of browser poll frequency.
    // The fixed ceiling leaves time for operator stills on the same listener.
    const MINIMUM_PLACEMENT_INTERVAL: Duration = Duration::from_millis(250);
    if !gateway.inner.setup_enabled.load(Ordering::Acquire) {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Live calibration is disabled",
        );
    }
    let Some(preview) = gateway
        .inner
        .frame_preview_stream
        .as_ref()
        .filter(|settings| settings.calibration.is_some())
    else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Calibration preview is unavailable",
        );
    };
    let Ok(mut last_request) = gateway.inner.ball_placement_last_request.try_lock() else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Placement preview is busy");
    };
    if last_request.is_some_and(|previous| previous.elapsed() < MINIMUM_PLACEMENT_INTERVAL) {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Placement preview is busy");
    }
    *last_request = Some(Instant::now());
    let request_gateway = gateway.clone();
    let request = tokio::task::spawn_blocking(move || -> anyhow::Result<PlacementRequest> {
        let authority = request_gateway
            .inner
            .configuration_authority
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Configuration is unavailable"))?;
        let configuration = authority
            .effective_configuration()
            .map_err(anyhow::Error::msg)?;
        let root = request_gateway
            .inner
            .config
            .project_root
            .clone()
            .map_or_else(std::env::current_dir, Ok)?;
        let capture = precision_capture(&configuration, &root)?;
        let expected = &capture.expected;
        let mut settings = serde_json::from_value(
            configuration["perception"]["rust_live"]["ball_localization"].clone(),
        )?;
        crate::browser_precision_calibration::select_assets(
            &mut settings,
            &root,
            expected.camera_profile_id(),
        );
        let context = crate::browser_precision_calibration::FitContext {
            root: &root,
            capture: &capture,
            settings,
            tokens: &Default::default(),
            maximum_jpeg_bytes: 0,
        };
        let optics = context.load_optics()?;
        let table = expected
            .calibration()
            .ok_or_else(|| anyhow::anyhow!("Calibrate table geometry first."))?;
        let target = BallSearchTarget::project(
            table,
            &optics.intrinsics,
            &context.settings.parameters,
            [query.x_mm, query.y_mm],
        )?;
        Ok(PlacementRequest {
            target,
            intrinsics: optics.intrinsics,
            parameters: context.settings.parameters,
            camera_profile_id: expected.camera_profile_id().to_owned(),
            configuration_revision: expected.configuration_revision(),
            calibration_revision: expected.calibration_revision(),
            rectification_revision: expected.rectification_revision(),
        })
    })
    .await;
    let request = match request {
        Ok(Ok(request)) => request,
        Ok(Err(error)) => return setup_failure(StatusCode::UNPROCESSABLE_ENTITY, error),
        Err(error) => return setup_failure(StatusCode::SERVICE_UNAVAILABLE, error),
    };
    let Ok(_preview) = gateway.inner.frame_preview_requests.try_read() else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Placement preview is busy");
    };
    let observation = request_ball_placement(preview, &request)
        .await
        .filter(|observation| request.matches(&observation.frame));
    let observation_available = observation.is_some();
    let ball_center = observation
        .and_then(|observation| observation.candidate)
        .map(|candidate| candidate.image_center_px);
    (
        [(CACHE_CONTROL, "no-store")],
        Json(json!({
            "width_px": request.intrinsics.source_width_px,
            "height_px": request.intrinsics.source_height_px,
            "target_center_px": request.target.center,
            "target_radius_px": request.target.tolerance,
            "observation_available": observation_available,
            "ball_center_px": ball_center,
            "in_target": ball_center.is_some_and(|center| request.target.contains(center)),
        })),
    )
        .into_response()
}

async fn request_ball_placement(
    preview: &FramePreviewStreamSettings,
    request: &PlacementRequest,
) -> Option<PlacementObservation> {
    tokio::time::timeout(preview.request_timeout, async {
        let bytes = serde_json::to_vec(request).ok()?;
        if bytes.len() > MAXIMUM_PLACEMENT_MESSAGE_BYTES {
            return None;
        }
        let mut stream = TcpStream::connect(preview.address).await.ok()?;
        stream.write_all(BALL_PLACEMENT_REQUEST_V1).await.ok()?;
        stream.write_u32(bytes.len() as u32).await.ok()?;
        stream.write_all(&bytes).await.ok()?;
        let length = stream.read_u32().await.ok()? as usize;
        if length == 0 || length > MAXIMUM_PLACEMENT_MESSAGE_BYTES {
            return None;
        }
        let mut response = vec![0_u8; length];
        stream.read_exact(&mut response).await.ok()?;
        serde_json::from_slice(&response).ok().flatten()
    })
    .await
    .ok()
    .flatten()
}

async fn setup_precision_operation(
    gateway: Gateway,
    reload_producer: bool,
    operation: impl FnOnce(crate::browser_precision_calibration::FitContext<'_>) -> anyhow::Result<Value>
        + Send
        + 'static,
) -> Response {
    if !gateway.inner.setup_enabled.load(Ordering::Acquire) {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Live calibration is disabled",
        );
    }
    let Some(maximum_jpeg_bytes) = gateway
        .inner
        .frame_preview_stream
        .as_ref()
        .filter(|settings| settings.calibration.is_some())
        .map(|settings| settings.maximum_jpeg_bytes)
    else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Calibration preview is unavailable",
        );
    };
    let Ok(_permit) = gateway
        .inner
        .configuration_slots
        .clone()
        .try_acquire_owned()
    else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Calibration is busy");
    };
    let Ok(_lock) = gateway
        .inner
        .configuration_mutation_lock
        .clone()
        .try_lock_owned()
    else {
        return setup_failure(StatusCode::CONFLICT, "Another setup change is in progress");
    };
    let result = tokio::task::spawn_blocking(move || -> anyhow::Result<Value> {
        // Keep the mutation guard until the capture or fit operation completes.
        let _lock = _lock;
        let _permit = _permit;
        let authority = gateway
            .inner
            .configuration_authority
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Configuration is unavailable"))?;
        let configuration = authority
            .effective_configuration()
            .map_err(anyhow::Error::msg)?;
        let root = gateway
            .inner
            .config
            .project_root
            .clone()
            .map_or_else(std::env::current_dir, Ok)?;
        let capture = precision_capture(&configuration, &root)?;
        let mut settings = serde_json::from_value(
            configuration["perception"]["rust_live"]["ball_localization"].clone(),
        )?;
        crate::browser_precision_calibration::select_assets(
            &mut settings,
            &root,
            capture.expected.camera_profile_id(),
        );
        let tokens = lock_unpoisoned(&gateway.inner.calibration_stills).clone();
        let result = operation(crate::browser_precision_calibration::FitContext {
            root: &root,
            capture: &capture,
            settings,
            tokens: &tokens,
            maximum_jpeg_bytes,
        })?;
        if reload_producer {
            gateway.inner.setup_reload.send_replace(());
        }
        Ok(result)
    })
    .await;
    match result {
        Ok(Ok(report)) => ([(CACHE_CONTROL, "no-store")], Json(report)).into_response(),
        Ok(Err(error)) => setup_failure(StatusCode::UNPROCESSABLE_ENTITY, error),
        Err(error) => setup_failure(StatusCode::SERVICE_UNAVAILABLE, error),
    }
}

async fn setup_calibration_apply(
    State(gateway): State<Gateway>,
    Json(request): Json<Value>,
) -> Response {
    match tokio::spawn(save_setup_calibration(gateway, request)).await {
        Ok(response) => response,
        Err(error) => setup_failure(StatusCode::SERVICE_UNAVAILABLE, error),
    }
}

async fn save_setup_calibration(gateway: Gateway, request: Value) -> Response {
    if !gateway.inner.setup_enabled.load(Ordering::Acquire) {
        return setup_failure(StatusCode::SERVICE_UNAVAILABLE, "Live capture is disabled");
    }
    let Some(settings) = gateway.inner.frame_preview_stream.as_ref() else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Calibration preview is disabled",
        );
    };
    let Some(limits) = settings.calibration.as_ref() else {
        return setup_failure(StatusCode::SERVICE_UNAVAILABLE, "Calibration is disabled");
    };
    let body = serde_json::to_vec(&request).expect("a JSON value can be serialized");
    if body.len() > limits.maximum_request_bytes {
        return setup_failure(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Calibration request is too large",
        );
    }
    let Ok(_permit) = gateway
        .inner
        .configuration_slots
        .clone()
        .try_acquire_owned()
    else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Calibration is busy");
    };
    let _guard = gateway.inner.configuration_mutation_lock.lock().await;
    let Some(authority) = gateway.inner.configuration_authority.clone() else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Setup storage is unavailable",
        );
    };
    let Ok(Ok(configuration)) =
        tokio::task::spawn_blocking(move || authority.effective_configuration()).await
    else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Unable to read setup configuration",
        );
    };
    // A source write may have committed while its old producer is still
    // draining. Do not allow it to save calibration for the superseded source.
    let expected_revision = configuration["metadata"]["configuration_revision"].as_u64();
    if expected_revision.is_none()
        || request["descriptor"]["configuration_revision"].as_u64() != expected_revision
    {
        return setup_failure(
            StatusCode::CONFLICT,
            "Camera setup changed. Take a new frame.",
        );
    }
    let _preview = gateway.inner.frame_preview_requests.write().await;
    let Ok(response) = crate::calibration_control::apply_gateway_calibration(
        settings.address,
        settings.request_timeout,
        limits.maximum_response_bytes,
        &body,
    )
    .await
    else {
        return setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Calibration could not be confirmed. Refresh setup before retrying.",
        );
    };
    if response["status"] != "ok" {
        return calibration_control_failure(response);
    }
    gateway.inner.setup_reload.send_replace(());
    Json(response).into_response()
}

fn calibration_control_failure(response: Value) -> Response {
    let status = match response["status"].as_str() {
        Some("stale_context" | "stale_revision") => StatusCode::CONFLICT,
        Some("invalid_request") => StatusCode::BAD_REQUEST,
        Some("busy") => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    };
    (status, Json(response)).into_response()
}

async fn setup_cameras(State(gateway): State<Gateway>) -> Response {
    let Ok(_permit) = gateway
        .inner
        .configuration_slots
        .clone()
        .try_acquire_owned()
    else {
        return setup_failure(StatusCode::TOO_MANY_REQUESTS, "Camera discovery is busy");
    };
    let executable = match std::env::current_exe() {
        Ok(path) => path.with_file_name("billiards-perception-camera-discovery"),
        Err(error) => return setup_failure(StatusCode::SERVICE_UNAVAILABLE, error),
    };
    let result = tokio::time::timeout(
        gateway.inner.config.tcp_read_timeout,
        tokio::process::Command::new(executable)
            .kill_on_drop(true)
            .output(),
    )
    .await;
    match result {
        Ok(Ok(output)) if output.status.success() => {
            match serde_json::from_slice::<Value>(&output.stdout) {
                Ok(report) => Json(report).into_response(),
                Err(error) => setup_failure(StatusCode::BAD_GATEWAY, error),
            }
        }
        _ => setup_failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Camera discovery is unavailable",
        ),
    }
}

async fn supervise_setup(
    gateway: Gateway,
    options: LiveProcessLaunchOptions,
    scene_address: SocketAddr,
    diagnostics_address: Option<SocketAddr>,
) {
    let mut shutdown = gateway.inner.shutdown.subscribe();
    let mut reload = gateway.inner.setup_reload.subscribe();
    let resolve = || -> Result<Option<LiveProcessConfig>, String> {
        let authority = gateway
            .inner
            .configuration_authority
            .as_ref()
            .ok_or("Live setup requires a configuration storage path")?;
        let configuration = authority.effective_configuration()?;
        if configured_setup_source(&configuration).is_none() {
            return Ok(None);
        }
        let project_root = gateway
            .inner
            .config
            .project_root
            .clone()
            .map_or_else(std::env::current_dir, Ok)
            .map_err(|error| error.to_string())?;
        let mut process = LiveProcessConfig::from_effective_configuration(
            &configuration,
            &project_root,
            scene_address,
            options.executable_path.as_deref(),
            options.capture_helper_path.as_deref(),
            options.audit_config,
        )?;
        if let Some(address) = diagnostics_address {
            process = process.with_diagnostics(
                address,
                gateway.inner.diagnostics_policy.clone(),
                gateway.inner.diagnostics.clone(),
                &configuration,
            )?;
        }
        Ok(Some(process))
    };
    while !*shutdown.borrow() {
        let process = match resolve() {
            Ok(process) => {
                *lock_unpoisoned(&gateway.inner.setup_error) = None;
                process
            }
            Err(error) => {
                *lock_unpoisoned(&gateway.inner.setup_error) = Some(error);
                None
            }
        };
        let child = process.map(|process| {
            let (stop, stopped) = watch::channel(false);
            let error_gateway = gateway.clone();
            (
                stop,
                tokio::spawn(process.supervise_with_failure(stopped, move |error| {
                    *lock_unpoisoned(&error_gateway.inner.setup_error) = Some(error);
                })),
            )
        });
        tokio::select! {
            _ = shutdown.changed() => {},
            _ = reload.changed() => {},
        }
        if let Some((stop, child)) = child {
            stop.send_replace(true);
            let _ = child.await;
            // Drop the outgoing scene immediately, even if the new source
            // takes time to produce its first frame.
            let mut state = lock_unpoisoned(&gateway.inner.state);
            state.updated_at = None;
            state.localization_received_at = None;
            state.live_source_geometry = None;
            state.live_configuration_revision = None;
            gateway.inner.latest.send_replace(None);
        }
    }
}

/// Start the gateway and optionally supervise the capture and inference producer.
pub async fn run_with_live(
    config: ServerConfig,
    live_options: LiveProcessLaunchOptions,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    config.validate().map_err(std::io::Error::other)?;
    let gateway_audit_exporter = match live_options.gateway_audit_config.as_ref() {
        Some(audit_config) => {
            audit_config.validate().map_err(std::io::Error::other)?;
            Some(LiveAuditExporter::start(audit_config.exporter).map_err(std::io::Error::other)?)
        }
        None => None,
    };
    let gateway_audit = gateway_audit_exporter.as_ref().map(|exporter| {
        let run_id = live_options
            .gateway_audit_config
            .as_ref()
            .expect("gateway exporter requires gateway audit config")
            .run_id
            .clone();
        (exporter.recorder(), run_id)
    });
    // Resolve listener policy at process startup, not in Gateway::new: HTTP
    // handlers must remain able to report corrupt/locked store failures.
    let diagnostics_policy = match config.configuration_path.as_ref() {
        Some(path) => {
            let authority = ConfigurationAuthority::new(path, AuthorityLimits::default())
                .map_err(std::io::Error::other)?;
            DiagnosticsConfig::from_configuration(
                &authority
                    .effective_configuration()
                    .map_err(std::io::Error::other)?,
            )
            .map_err(std::io::Error::other)?
        }
        None => DiagnosticsConfig::default(),
    };
    let gateway =
        Gateway::new_with_audit_and_diagnostics(config.clone(), gateway_audit, diagnostics_policy)
            .map_err(std::io::Error::other)?;
    let scene_listener =
        TcpListener::bind(SocketAddr::new(config.scene_bind, config.scene_port)).await?;
    let http_listener =
        TcpListener::bind(SocketAddr::new(config.http_bind, config.http_port)).await?;
    let scene_address = scene_listener.local_addr()?;
    let http_address = http_listener.local_addr()?;
    eprintln!("billiards Rust gateway HTTP listening on {http_address}");
    eprintln!("live v3 scene ingress listening on {scene_address}");
    let diagnostics_listener = if gateway.inner.diagnostics_policy.enabled {
        Some(
            TcpListener::bind(SocketAddr::new(
                gateway.inner.diagnostics_policy.bind_address,
                gateway.inner.diagnostics_policy.port,
            ))
            .await?,
        )
    } else {
        None
    };
    let diagnostics_address = diagnostics_listener
        .as_ref()
        .map(TcpListener::local_addr)
        .transpose()?;
    gateway
        .inner
        .setup_enabled
        .store(live_options.enabled, Ordering::Release);
    let diagnostics_ingress = diagnostics_listener.map(|listener| {
        tokio::spawn(crate::producer_diagnostics::serve(
            listener,
            gateway.inner.diagnostics.clone(),
            gateway.inner.diagnostics_policy.clone(),
            gateway.inner.config.maximum_websocket_clients,
            gateway.inner.shutdown.subscribe(),
        ))
    });
    let ingress_gateway = gateway.clone();
    let ingress = tokio::spawn(async move {
        ingress_gateway
            .accept_scene_connections(scene_listener)
            .await
    });
    let freshness = tokio::spawn(run_freshness_scheduler(gateway.clone()));
    let diagnostics_publisher = tokio::spawn(run_diagnostics_snapshot_publisher(gateway.clone()));
    let live_supervisor = live_options.enabled.then(|| {
        tokio::spawn(supervise_setup(
            gateway.clone(),
            live_options,
            scene_address,
            diagnostics_address,
        ))
    });
    let shutdown_gateway = gateway.clone();
    let server = axum::serve(http_listener, gateway.router()).with_graceful_shutdown(async move {
        shutdown_signal().await;
        shutdown_gateway.shutdown();
    });
    let server_result = server.await;
    gateway.shutdown();
    freshness.abort();
    diagnostics_publisher.abort();
    let _ = diagnostics_publisher.await;
    if let Some(diagnostics_ingress) = diagnostics_ingress {
        diagnostics_ingress.await??;
    }
    if let Some(live_supervisor) = live_supervisor {
        live_supervisor.await?;
    }
    match ingress.await {
        Ok(result) => result?,
        Err(error) if error.is_cancelled() => {}
        Err(error) => return Err(error.into()),
    }
    server_result?;
    if let Some(exporter) = gateway_audit_exporter {
        let summary = exporter.finish();
        eprintln!(
            "gateway_audit_export_summary={}",
            serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_owned())
        );
    }
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c()
        .await
        .expect("install Ctrl-C handler");
}

impl Gateway {
    async fn accept_scene_connections(&self, listener: TcpListener) -> Result<(), std::io::Error> {
        let mut shutdown = self.inner.shutdown.subscribe();
        loop {
            let (stream, peer) = tokio::select! {
                accepted = listener.accept() => accepted?,
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow_and_update() {
                        return Ok(());
                    }
                    continue;
                }
            };
            if !peer.ip().is_loopback() {
                drop(stream);
                continue;
            }
            if self
                .inner
                .producer_connected
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                drop(stream);
                continue;
            }
            self.inner.producer_connection_updates.send_replace(true);
            let gateway = self.clone();
            let standard_stream = match stream.into_std() {
                Ok(stream) => stream,
                Err(_) => {
                    self.inner
                        .producer_connected
                        .store(false, Ordering::Release);
                    self.inner.producer_connection_updates.send_replace(false);
                    continue;
                }
            };
            let control_stream = match standard_stream.try_clone() {
                Ok(stream) => stream,
                Err(_) => {
                    self.inner
                        .producer_connected
                        .store(false, Ordering::Release);
                    self.inner.producer_connection_updates.send_replace(false);
                    continue;
                }
            };
            *lock_unpoisoned(&self.inner.producer_control_stream) = Some(control_stream);
            tokio::task::spawn_blocking(move || {
                let _active = ActiveProducerGuard(gateway.inner.clone());
                if let Err(error) = gateway.handle_producer(standard_stream) {
                    eprintln!("scene producer connection closed: {error}");
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{header::CONTENT_TYPE, Request};
    use futures_util::StreamExt;
    use serde_json::Value;
    use socket2::{Domain, Protocol, Socket, Type};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
    use tokio_tungstenite::tungstenite::Message as ClientMessage;
    use tokio_tungstenite::{client_async_with_config, connect_async_with_config};
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
    use tower::ServiceExt;

    type TestWebSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

    fn diagnostics_test_policy() -> DiagnosticsConfig {
        DiagnosticsConfig {
            heartbeat_interval_ms: 20,
            report_expiry_ms: 100,
            stream_heartbeat_interval_ms: 20,
            stream_stale_after_ms: 80,
            processing_stall_timeout_ms: 50,
            ..DiagnosticsConfig::default()
        }
    }

    async fn receive_diagnostics_snapshot(client: &mut TestWebSocket) -> Value {
        let message = tokio::time::timeout(Duration::from_secs(1), client.next())
            .await
            .expect("timed out waiting for diagnostics snapshot")
            .expect("diagnostics websocket closed")
            .expect("diagnostics websocket receive failed");
        let ClientMessage::Text(encoded) = message else {
            panic!("expected full diagnostics JSON snapshot")
        };
        serde_json::from_str(&encoded).expect("diagnostics message must be valid JSON")
    }

    fn test_config() -> ServerConfig {
        ServerConfig {
            http_port: 0,
            scene_port: 0,
            tcp_read_timeout: Duration::from_millis(250),
            stale_after: Duration::from_millis(100),
            expire_after: Duration::from_millis(500),
            ..ServerConfig::default()
        }
    }

    #[test]
    fn setup_reports_missing_precision_assets_and_rejects_invalid_optics() {
        let (directory, configuration_path) = live_test_configuration();
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(configuration_path).unwrap()).unwrap();
        let missing = setup_precision_calibration(&config, directory.path());
        assert_eq!(missing["optics"]["state"], "missing");
        assert_eq!(missing["ball_centers"]["state"], "missing");

        std::fs::write(directory.path().join("invalid-optics.json"), "not json").unwrap();
        config["perception"]["rust_live"]["ball_localization"]["optics_profile_path"] =
            json!("invalid-optics.json");
        let invalid = setup_precision_calibration(&config, directory.path());
        assert_eq!(invalid["optics"]["state"], "invalid");
        assert_eq!(invalid["ball_centers"]["state"], "missing");
    }

    #[test]
    fn setup_recognizes_table_calibration_without_precise_ball_coordinates() {
        let mut status = json!({
            "calibration_state": "unavailable",
            "perception": {"table_visibility": {"reason": "reference_match"}}
        });
        assert!(setup_table_calibrated(&status));
        status["perception"]["table_visibility"]["reason"] = json!("cloth_mismatch");
        assert!(setup_table_calibrated(&status));
        status["perception"]["table_visibility"]["reason"] = json!("missing_reference");
        assert!(!setup_table_calibrated(&status));
    }

    #[tokio::test]
    async fn setup_is_ready_with_visibility_before_ball_localization() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.json");
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(path),
            project_root: Some(directory.path().to_owned()),
            ..test_config()
        })
        .unwrap();
        gateway.inner.setup_enabled.store(true, Ordering::Release);
        let source = setup_source(
            State(gateway.clone()),
            Json(SetupSource {
                expected_revision: 1,
                source: "http://localhost:8001/stream".to_owned(),
            }),
        )
        .await;
        let bytes = to_bytes(source.into_body(), 4096).await.unwrap();
        let revision = serde_json::from_slice::<Value>(&bytes).unwrap()["configuration_revision"]
            .as_u64()
            .unwrap();
        gateway
            .activate_producer_epoch("engine".to_owned(), "session".to_owned())
            .unwrap();
        gateway
            .inner
            .producer_connected
            .store(true, Ordering::Release);
        {
            let mut state = lock_unpoisoned(&gateway.inner.state);
            state.live_configuration_revision = Some(revision);
            state.updated_at = Some(Utc::now());
            state
                .lifecycle
                .update_perception_status(
                    "engine",
                    "session",
                    json!({"table_visibility":{"reason":"reference_match"}}),
                    0,
                    0,
                )
                .unwrap();
        }
        let status = setup_status(State(gateway.clone())).await;
        let bytes = to_bytes(status.into_body(), 4096).await.unwrap();
        let status: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status["state"], "running");
        assert_eq!(status["calibrated"], true);
        lock_unpoisoned(&gateway.inner.state).live_configuration_revision = Some(revision - 1);
        let status = setup_status(State(gateway)).await;
        let bytes = to_bytes(status.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["state"],
            "starting"
        );
    }

    #[tokio::test]
    async fn first_camera_selection_is_sparse_revisioned_and_requests_reload() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("config.json");
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(path.clone()),
            project_root: Some(directory.path().to_owned()),
            ..test_config()
        })
        .unwrap();
        gateway.inner.setup_enabled.store(true, Ordering::Release);
        let mut reload = gateway.inner.setup_reload.subscribe();
        let response = setup_status(State(gateway.clone())).await;
        let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["state"],
            "idle"
        );
        assert!(!path.exists());

        let source = "http://127.0.0.1:9000/camera.mjpg";
        let response = setup_source(
            State(gateway.clone()),
            Json(SetupSource {
                expected_revision: 1,
                source: source.to_owned(),
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(reload.has_changed().unwrap());
        reload.borrow_and_update();
        let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(saved["vision"]["camera"]["device_id"], source);
        assert!(saved.get("api").is_none());
        assert_eq!(saved["metadata"]["configuration_revision"], 2);

        let stale = setup_source(
            State(gateway),
            Json(SetupSource {
                expected_revision: 1,
                source: "http://127.0.0.1:9001/other.mjpg".to_owned(),
            }),
        )
        .await;
        assert_eq!(stale.status(), StatusCode::CONFLICT);
        assert!(!reload.has_changed().unwrap_or(false));
    }

    fn live_test_configuration() -> (tempfile::TempDir, PathBuf) {
        let mut document: Value =
            serde_json::from_str(include_str!("../contracts/configuration/v1/defaults.json"))
                .unwrap();
        document["vision"]["camera"]["video_file_path"] = json!("");
        document["vision"]["camera"]["resolution"] = json!([3840, 2160]);
        let directory = tempfile::tempdir().unwrap();
        document["game_session"]["store_path"] = json!(directory
            .path()
            .join("game-session.json")
            .to_string_lossy()
            .to_string());
        let path = directory.path().join("config.json");
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        (directory, path)
    }

    #[tokio::test]
    async fn calibration_route_rejects_a_frame_from_an_old_configuration() {
        let (_directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            ..test_config()
        })
        .unwrap();
        gateway.inner.setup_enabled.store(true, Ordering::Release);
        let reload = gateway.inner.setup_reload.subscribe();

        let response = gateway
            .router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/setup/calibration")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"descriptor":{"configuration_revision":0}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert!(!reload.has_changed().unwrap());
    }

    fn test_calibration_capture(
        capture: &ResolvedCaptureSettings,
        jpeg: &[u8],
    ) -> (FramePreviewDescriptorV1, Value, (String, [u8; 32])) {
        let expected = &capture.expected;
        let geometry = expected.output_frame_geometry().unwrap();
        let descriptor: FramePreviewDescriptorV1 = serde_json::from_value(json!({
            "schema_version": 1,
            "engine_instance_id": "test-engine",
            "capture_session_id": "test-session",
            "source_frame_id": "1",
            "width_px": geometry.width_px(),
            "height_px": geometry.height_px(),
            "pixel_format": "RGB8",
            "image_space": "raw",
            "camera_profile_id": expected.camera_profile_id(),
            "configuration_revision": expected.configuration_revision(),
            "calibration_revision": expected.calibration_revision(),
            "rectification_revision": expected.rectification_revision(),
            "source_timestamp": null,
            "source_timestamp_semantics": {"kind":"read_completion_only"},
            "host_read_completed": {
                "ticks": "1", "unit": "nanoseconds", "clock": "monotonic",
                "domain_id": "test-domain", "stage": "camera_read_complete",
                "semantics": "read_completion"
            },
        }))
        .unwrap();
        let token = "test-capture".to_owned();
        let digest = crate::browser_precision_calibration::fingerprint(&descriptor, jpeg);
        let body = json!({
            "descriptor": descriptor,
            "jpeg_base64": base64::engine::general_purpose::STANDARD.encode(jpeg),
            "capture_token": token,
        });
        (descriptor, body, (token, digest))
    }

    #[tokio::test]
    async fn setup_archives_calibration_thumbnails_across_gateway_restart() {
        let (directory, configuration_path) = live_test_configuration();
        let new_gateway = || {
            let gateway = Gateway::new(ServerConfig {
                configuration_path: Some(configuration_path.clone()),
                project_root: Some(directory.path().to_owned()),
                ..test_config()
            })
            .unwrap();
            gateway.inner.setup_enabled.store(true, Ordering::Release);
            gateway
        };
        let gateway = new_gateway();
        let configuration = gateway
            .inner
            .configuration_authority
            .as_ref()
            .unwrap()
            .effective_configuration()
            .unwrap();
        let capture = precision_capture(&configuration, directory.path()).unwrap();
        let geometry = capture.expected.output_frame_geometry().unwrap();
        let image = image::RgbImage::from_pixel(
            geometry.width_px(),
            geometry.height_px(),
            image::Rgb([15, 120, 80]),
        );
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode_image(&image)
            .unwrap();
        let (_, capture_body, token) = test_calibration_capture(&capture, &jpeg);
        lock_unpoisoned(&gateway.inner.calibration_stills).push_back(token);
        let route = "/api/v1/setup/calibration/captures/optics";
        let request = |method, slot: Option<usize>, body: Option<Value>| {
            Request::builder()
                .method(method)
                .uri(slot.map_or_else(|| route.to_owned(), |slot| format!("{route}/{slot}")))
                .header(CONTENT_TYPE, "application/json")
                .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
                .unwrap()
        };
        let response = gateway
            .router()
            .oneshot(request("PUT", Some(3), Some(capture_body.clone())))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
        let saved: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 512 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(saved["slot"], 3);
        let response = gateway
            .router()
            .oneshot(request("PUT", Some(1), Some(capture_body)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let restart = new_gateway();
        assert!(lock_unpoisoned(&restart.inner.calibration_stills).is_empty());
        let response = restart
            .router()
            .oneshot(request("GET", None, None))
            .await
            .unwrap();
        assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
        let listed: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 512 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(listed["captures"], json!([saved]));
        assert!(listed["captures"][0].get("jpeg_base64").is_none());

        let response = restart
            .router()
            .oneshot(request("DELETE", None, None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = restart
            .router()
            .oneshot(request("GET", None, None))
            .await
            .unwrap();
        let listed: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(listed["captures"], json!([]));
    }

    #[test]
    fn setup_ball_candidate_refines_a_rendered_jpeg_near_the_measured_position() {
        use crate::ball_localization::{BallLocalizationConfig, CameraIntrinsics};
        use crate::ball_localization_profile::{
            BallLocalizationSettings, CameraOpticsProfileV1, BALL_REFINER_REVISION,
            CAMERA_OPTICS_PROFILE_VERSION,
        };
        use crate::browser_precision_calibration::{asset_paths, save_asset, FitContext};
        use crate::calibration::{
            solve_table_calibration, ImagePointPx, TableCalibrationSolverInput,
        };
        let (directory, configuration_path) = live_test_configuration();
        let mut configuration: Value =
            serde_json::from_slice(&std::fs::read(configuration_path).unwrap()).unwrap();
        configuration["vision"]["camera"]["resolution"] = json!([480, 300]);
        let mut capture = precision_capture(&configuration, directory.path()).unwrap();
        let profile = capture.expected.camera_profile_id().to_owned();
        let table = solve_table_calibration(TableCalibrationSolverInput {
            calibration_revision: 1,
            camera_profile_id: profile.clone(),
            rectification_revision: capture.expected.rectification_revision(),
            image_space: crate::ImageSpace::Raw,
            source_width_px: 480,
            source_height_px: 300,
            table_width_mm: 600.0,
            table_height_mm: 350.0,
            corners_px: [[40.0, 34.0], [440.0, 34.0], [440.0, 266.0], [40.0, 266.0]]
                .map(|[x_px, y_px]| ImagePointPx { x_px, y_px }),
        })
        .unwrap();
        capture
            .expected
            .apply_calibration_store_state(1, Some(table.into_validated().unwrap()))
            .unwrap();
        let optics_path = asset_paths(directory.path(), &profile).0;
        save_asset(
            &optics_path,
            &CameraOpticsProfileV1 {
                schema_version: CAMERA_OPTICS_PROFILE_VERSION,
                optics_profile_id: "test-optics".into(),
                camera_profile_id: profile,
                optics_revision: 1,
                refiner_revision: BALL_REFINER_REVISION,
                image_space: crate::ImageSpace::Raw,
                intrinsics: CameraIntrinsics {
                    source_width_px: 480,
                    source_height_px: 300,
                    fx: 260.0,
                    fy: 260.0,
                    cx: 240.0,
                    cy: 150.0,
                },
                provenance: "Synthetic calibration fixture".into(),
            },
        )
        .unwrap();
        let ball_center = [240.0, 150.0];
        let image = image::RgbImage::from_fn(480, 300, |x, y| {
            let distance = (f64::from(x) - ball_center[0]).hypot(f64::from(y) - ball_center[1]);
            let alpha = (19.5 - distance).clamp(0.0, 1.0);
            let cloth = [24.0, 145.0, 118.0];
            let ball = [225.0, 62.0, 35.0];
            image::Rgb(std::array::from_fn(|ch| {
                ((1.0 - alpha) * cloth[ch] + alpha * ball[ch]) as u8
            }))
        });
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 92)
            .encode_image(&image)
            .unwrap();
        let (_, mut body, token) = test_calibration_capture(&capture, &jpeg);
        body["table_center_mm"] = json!([0.0, 0.0]);
        let report = FitContext {
            root: directory.path(),
            capture: &capture,
            settings: BallLocalizationSettings {
                optics_profile_path: Some(optics_path),
                parameters: BallLocalizationConfig {
                    reference_width_px: 480,
                    batch_budget_ms: 1000,
                    proposal_budget_ms: 500,
                    ..Default::default()
                },
                ..Default::default()
            },
            tokens: &std::collections::VecDeque::from([token]),
            maximum_jpeg_bytes: jpeg.len(),
        }
        .ball_candidate(serde_json::from_value(body).unwrap())
        .unwrap();
        let measured: [f64; 2] =
            serde_json::from_value(report["candidate"]["image_center_px"].clone())
                .unwrap_or_else(|_| panic!("No refined candidate: {report}"));
        assert!((measured[0] - ball_center[0]).hypot(measured[1] - ball_center[1]) < 1.0);
        assert!((report["candidate"]["radius_px"].as_f64().unwrap() - 19.0).abs() < 1.0);
    }

    #[tokio::test]
    async fn setup_serves_the_checked_in_checkerboard_as_a_pdf_download() {
        let response = Gateway::new(test_config())
            .unwrap()
            .router()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/setup/calibration/checkerboard.pdf")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/pdf");
        assert!(response.headers()[axum::http::header::CONTENT_DISPOSITION]
            .to_str()
            .unwrap()
            .starts_with("attachment;"));
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        assert_eq!(
            body.as_ref(),
            include_bytes!("../../output/pdf/ball-localization-checkerboard-letter-tiles.pdf")
        );
    }

    #[tokio::test]
    async fn gateway_has_no_video_stream_route() {
        let response = Gateway::new(test_config())
            .unwrap()
            .router()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/video/identified.mjpg")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn verified_empty_state_is_serialized_and_required_by_delivery_schema() {
        let state = PerceptionStateV1 {
            event_sequence: 7,
            image_width_px: Some(1280),
            image_height_px: Some(720),
            phase: "pre_shot".into(),
            layout_revision: 3,
            layout_trusted: true,
            layout_basis_frame_id: Some("frame-1".into()),
            empty_layout_verified: true,
            guidance_valid: false,
            trust_reason: "trusted".into(),
            last_ball_observation_age_ms: Some(0),
            latest_motion_check_age_ms: Some(0),
            cue_evidence_age_ms: None,
            motion_evidence_max_age_ms: 200,
            cue_evidence_max_age_ms: 250,
            layout_verification_max_age_ms: 1_500,
            cue: None,
            decision_frame_id: None,
            decision_calibration_revision: None,
            decision_rectification_revision: None,
            table_visibility: None,
            table_projection: None,
        };
        assert!(validate_perception_state(&state).is_ok());
        let encoded = serde_json::to_value(&state).unwrap();
        assert_eq!(encoded["empty_layout_verified"], true);

        let schema: Value = serde_json::from_str(include_str!(
            "../contracts/public-state/public-state-delivery-v2.schema.json"
        ))
        .unwrap();
        let perception_schema = &schema["$defs"]["perception"];
        assert!(perception_schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "empty_layout_verified"));
        assert_eq!(
            perception_schema["properties"]["empty_layout_verified"]["type"],
            "boolean"
        );
        assert_eq!(
            perception_schema["properties"]["layout_basis_frame_id"]["type"],
            json!(["string", "null"])
        );
    }

    #[test]
    fn verified_empty_state_requires_trust_and_current_motion_evidence() {
        let mut state = PerceptionStateV1 {
            event_sequence: 1,
            image_width_px: None,
            image_height_px: None,
            phase: "recovering".into(),
            layout_revision: 0,
            layout_trusted: false,
            layout_basis_frame_id: None,
            empty_layout_verified: true,
            guidance_valid: false,
            trust_reason: "awaiting_settlement".into(),
            last_ball_observation_age_ms: None,
            latest_motion_check_age_ms: Some(0),
            cue_evidence_age_ms: None,
            motion_evidence_max_age_ms: 200,
            cue_evidence_max_age_ms: 250,
            layout_verification_max_age_ms: 1_500,
            cue: None,
            decision_frame_id: None,
            decision_calibration_revision: None,
            decision_rectification_revision: None,
            table_visibility: None,
            table_projection: None,
        };
        assert!(validate_perception_state(&state).is_err());
        state.layout_trusted = true;
        state.last_ball_observation_age_ms = None;
        state.latest_motion_check_age_ms = Some(0);
        assert!(validate_perception_state(&state).is_ok());
    }

    #[test]
    fn guidance_requires_current_motion_and_cue_evidence_without_frame_join() {
        let mut state = PerceptionStateV1 {
            event_sequence: 2,
            image_width_px: Some(1280),
            image_height_px: Some(720),
            phase: "aiming".into(),
            layout_revision: 4,
            layout_trusted: true,
            layout_basis_frame_id: Some("frame-2".into()),
            empty_layout_verified: false,
            guidance_valid: true,
            trust_reason: "trusted".into(),
            last_ball_observation_age_ms: Some(0),
            latest_motion_check_age_ms: Some(0),
            cue_evidence_age_ms: Some(0),
            motion_evidence_max_age_ms: 200,
            cue_evidence_max_age_ms: 250,
            layout_verification_max_age_ms: 1_500,
            cue: Some(crate::public_state_lifecycle::CueGeometryStateV1 {
                shaft_start_px: [100.0, 200.0],
                shaft_end_px: [500.0, 180.0],
                axis_unit: [0.998_752, -0.049_938],
                confidence: 0.9,
                observation_age_ms: 0,
                signed_direction: None,
                source_frame_id: Some("frame-2".into()),
            }),
            decision_frame_id: Some("frame-2".into()),
            decision_calibration_revision: Some(7),
            decision_rectification_revision: Some(9),
            table_projection: None,
            table_visibility: Some(crate::public_state_lifecycle::TableVisibilityStateV1 {
                status: crate::public_state_lifecycle::TableVisibilityStatusV1::Visible,
                reason: "reference_match".into(),
                observation_frame_id: "frame-2".into(),
                observation_age_ms: 0,
            }),
        };
        assert!(validate_perception_state(&state).is_ok());

        state.table_visibility = None;
        assert!(validate_perception_state(&state).is_ok());
        state.table_visibility = Some(crate::public_state_lifecycle::TableVisibilityStateV1 {
            status: crate::public_state_lifecycle::TableVisibilityStatusV1::NotVisible,
            reason: "cloth_mismatch".into(),
            observation_frame_id: "frame-2".into(),
            observation_age_ms: 0,
        });
        assert!(validate_perception_state(&state).is_ok());
        state.table_visibility = Some(crate::public_state_lifecycle::TableVisibilityStateV1 {
            status: crate::public_state_lifecycle::TableVisibilityStatusV1::Visible,
            reason: "reference_match".into(),
            observation_frame_id: "frame-2".into(),
            observation_age_ms: 201,
        });
        assert!(validate_perception_state(&state).is_ok());
        state.table_visibility.as_mut().unwrap().observation_age_ms = 0;
        state.decision_frame_id = None;
        state.cue.as_mut().unwrap().source_frame_id = Some("older-cue-frame".into());
        state.last_ball_observation_age_ms = Some(9_000);
        assert!(validate_perception_state(&state).is_ok());
        state.latest_motion_check_age_ms = Some(201);
        assert!(validate_perception_state(&state).is_err());
    }

    fn live_acknowledgement(settings: &ResolvedCaptureSettings, session_id: &str) -> Value {
        let mut acknowledgement: Value = serde_json::from_str(include_str!(
            "../contracts/capture-session-ack/v2/capture-session-ack-default.json"
        ))
        .unwrap();
        let report = &mut acknowledgement["source_report"];
        report["session_id"] = json!(session_id);
        report["device_id"] = json!(settings.expected.device_id());
        report["requested_profile"] =
            serde_json::to_value(settings.expected.requested_profile()).unwrap();
        report["negotiated_profile"] =
            serde_json::to_value(settings.expected.expected_negotiated_profile()).unwrap();
        report["output_frame_geometry"] =
            serde_json::to_value(settings.expected.output_frame_geometry().unwrap()).unwrap();
        report["applied_frame_geometry"]["configuration_revision"] =
            json!(settings.configuration_revision);
        report["applied_frame_geometry"]["camera_profile_id"] =
            json!(settings.expected.camera_profile_id());
        report["applied_frame_geometry"]["image_space"] =
            serde_json::to_value(settings.expected.image_space()).unwrap();
        report["applied_frame_geometry"]["rectification_revision"] =
            json!(settings.expected.rectification_revision());
        acknowledgement["configuration_revision"] = json!(settings.configuration_revision);
        acknowledgement["request_sha256"] = json!(settings.request_sha256);
        acknowledgement
    }

    fn fixture_scene() -> SceneSnapshotV1 {
        crate::decode_scene_snapshot_v1(include_bytes!(
            "../contracts/v1/scene-rust-tracked-success.json"
        ))
        .unwrap()
    }

    async fn start_ingress(gateway: Gateway) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            gateway.accept_scene_connections(listener).await.unwrap();
        });
        address
    }

    async fn send_frame(stream: &mut TcpStream, value: &[u8]) {
        stream
            .write_all(&(value.len() as u32).to_be_bytes())
            .await
            .unwrap();
        stream.write_all(value).await.unwrap();
    }

    async fn read_frame(stream: &mut TcpStream) -> Vec<u8> {
        let mut header = [0; 4];
        stream.read_exact(&mut header).await.unwrap();
        let mut payload = vec![0; u32::from_be_bytes(header) as usize];
        stream.read_exact(&mut payload).await.unwrap();
        payload
    }

    fn scene_for(engine: &str, session: &str, sequence: u64, frame: &str) -> SceneSnapshotV1 {
        let mut scene = fixture_scene();
        scene.engine_instance_id = engine.into();
        scene.source.session_id = session.into();
        scene.source.frame_id = frame.into();
        scene.transport_sequence = sequence;
        scene.snapshot_revision = sequence;
        scene.source.timestamp.ticks = (1_000_000_000 + sequence).to_string();
        scene.published_at.ticks = (1_000_000_001 + sequence).to_string();
        for ball in &mut scene.balls {
            ball.last_observation.session_id = session.into();
            ball.last_observation.frame_id = frame.into();
            ball.last_observation.timestamp.ticks = scene.source.timestamp.ticks.clone();
        }
        scene
    }

    async fn start_websocket(
        gateway: &Gateway,
        query: &str,
    ) -> (tokio::task::JoinHandle<()>, TestWebSocket) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = gateway.router();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let url = format!("ws://{address}/api/v1/websocket/ws{query}");
        let (client, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        (server, client)
    }

    async fn receive_websocket_state(client: &mut TestWebSocket) -> Value {
        let message = tokio::time::timeout(Duration::from_secs(1), client.next())
            .await
            .expect("timed out waiting for WebSocket state")
            .expect("WebSocket closed before state delivery")
            .expect("WebSocket receive failed");
        let ClientMessage::Text(encoded) = message else {
            panic!("expected a WebSocket state message")
        };
        serde_json::from_str(&encoded).expect("server should send JSON state")
    }

    fn test_perception_state(
        event_sequence: u64,
        layout_trusted: bool,
        empty_layout_verified: bool,
    ) -> PerceptionStateV1 {
        PerceptionStateV1 {
            event_sequence,
            image_width_px: Some(1280),
            image_height_px: Some(720),
            phase: if layout_trusted {
                "pre_shot".into()
            } else {
                "recovering".into()
            },
            layout_revision: u64::from(layout_trusted),
            layout_trusted,
            layout_basis_frame_id: layout_trusted.then(|| "frame-1".to_owned()),
            empty_layout_verified,
            guidance_valid: false,
            trust_reason: if layout_trusted {
                "verified_layout".into()
            } else {
                "layout_unverified".into()
            },
            last_ball_observation_age_ms: Some(0),
            latest_motion_check_age_ms: Some(0),
            cue_evidence_age_ms: None,
            motion_evidence_max_age_ms: 200,
            cue_evidence_max_age_ms: 250,
            layout_verification_max_age_ms: 1_500,
            cue: None,
            decision_frame_id: None,
            decision_calibration_revision: None,
            decision_rectification_revision: None,
            table_visibility: None,
            table_projection: None,
        }
    }

    fn test_guidance_state(event_sequence: u64, guidance_valid: bool) -> PerceptionStateV1 {
        let mut state = test_perception_state(event_sequence, true, false);
        state.phase = "aiming".into();
        state.guidance_valid = guidance_valid;
        state.cue_evidence_age_ms = guidance_valid.then_some(0);
        if guidance_valid {
            state.cue = Some(crate::public_state_lifecycle::CueGeometryStateV1 {
                shaft_start_px: [100.0, 200.0],
                shaft_end_px: [500.0, 180.0],
                axis_unit: [0.998_752, -0.049_938],
                confidence: 0.9,
                observation_age_ms: 0,
                signed_direction: None,
                source_frame_id: Some("frame-guidance".into()),
            });
        }
        state
    }

    fn public_state_delivery(scene: &SceneSnapshotV1) -> Vec<u8> {
        let mut lifecycle = SceneStateLifecycle::default();
        lifecycle
            .activate_source(&scene.engine_instance_id, &scene.source.session_id)
            .unwrap();
        lifecycle.ingest(scene, 100, 100, None, None).unwrap();
        let current = lifecycle.current(100).unwrap();
        let delivery =
            crate::public_state_lifecycle::PublicStateDeliveryV1::new(scene, current.as_ref());
        serde_json::to_vec(&delivery).unwrap()
    }

    fn public_state_delivery_v2(
        scene: &SceneSnapshotV1,
        delivery_sequence: u64,
        perception: PerceptionStateV1,
    ) -> Vec<u8> {
        let legacy: Value = serde_json::from_slice(&public_state_delivery(scene)).unwrap();
        serde_json::to_vec(&json!({
            "delivery_type": "billiards_public_state_delivery",
            "delivery_schema_version": 2,
            "delivery_sequence": delivery_sequence,
            "engine_instance_id": scene.engine_instance_id,
            "source_session_id": scene.source.session_id,
            "source_snapshot": legacy["source_snapshot"],
            "current_state": legacy["current_state"],
            "perception": perception,
        }))
        .unwrap()
    }

    fn dense_scene(sequence: u64) -> SceneSnapshotV1 {
        let mut scene = scene_for(
            "engine-a",
            "session-a",
            sequence,
            &format!("frame-{sequence}"),
        );
        let template = scene.balls[0].clone();
        for index in 1..64 {
            let mut ball = template.clone();
            ball.track_id = format!("track-{}", index + 1);
            scene.balls.push(ball);
        }
        scene
    }

    fn oversized_public_state_scene(sequence: u64) -> SceneSnapshotV1 {
        let mut scene = dense_scene(sequence);
        for ball in &mut scene.balls {
            // Evidence is deliberately repetitive and bounded by the v1 contract's
            // per-identifier limit. It makes the real WebSocket write exceed local
            // socket buffering without introducing invalid coordinates or data.
            ball.identity.evidence = (0..2_048)
                .map(|_| crate::IdentityEvidence {
                    source: "e".repeat(120),
                    confidence: None,
                })
                .collect();
        }
        scene
    }

    fn mask_contract_time_and_sequence(value: &mut Value) {
        match value {
            Value::Object(object) => {
                for (key, child) in object.iter_mut() {
                    if key == "sequence" {
                        *child = json!(0);
                    } else if key == "timestamp" && child.is_string() {
                        *child = json!("<timestamp>");
                    } else {
                        mask_contract_time_and_sequence(child);
                    }
                }
            }
            Value::Array(items) => {
                for child in items {
                    mask_contract_time_and_sequence(child);
                }
            }
            _ => {}
        }
    }

    fn assert_object_keys(value: &Value, expected: &[&str]) {
        let actual = value
            .as_object()
            .expect("contract value should be an object")
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let expected = expected
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual, expected);
    }

    fn assert_fixture_fields(actual: &Value, expected: &Value, path: &str) {
        match expected {
            Value::Object(fields) => {
                let actual = actual
                    .as_object()
                    .unwrap_or_else(|| panic!("{path} should be an object"));
                for (key, expected) in fields {
                    let child_path = format!("{path}.{key}");
                    let actual = actual
                        .get(key)
                        .unwrap_or_else(|| panic!("missing fixture field {child_path}"));
                    assert_fixture_fields(actual, expected, &child_path);
                }
            }
            Value::Array(expected_items) => {
                let actual = actual
                    .as_array()
                    .unwrap_or_else(|| panic!("{path} should be an array"));
                assert_eq!(actual.len(), expected_items.len(), "{path} length");
                for (index, (actual, expected)) in actual.iter().zip(expected_items).enumerate() {
                    assert_fixture_fields(actual, expected, &format!("{path}[{index}]"));
                }
            }
            _ => match (actual, expected) {
                (Value::Number(actual), Value::Number(expected)) => assert_eq!(
                    actual.as_f64(),
                    expected.as_f64(),
                    "numeric contract value at {path}"
                ),
                _ => assert_eq!(actual, expected, "{path}"),
            },
        }
    }

    async fn get_json(router: Router, path: &str) -> (StatusCode, Value) {
        let response = router
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn gateway_composes_configured_static_fallback_after_api_routes() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("index.html"),
            "<main>rust client</main>",
        )
        .unwrap();
        let gateway = Gateway::new(ServerConfig {
            web_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let response = gateway
            .router()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(body.as_ref(), b"<main>rust client</main>");

        let missing_api = gateway
            .router()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing_api.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn validates_cli_runtime_configuration_and_requires_loopback_scene_bind() {
        assert!(ServerConfig::default().validate().is_ok());
        for invalid in [
            ServerConfig {
                scene_bind: "0.0.0.0".parse().unwrap(),
                ..test_config()
            },
            ServerConfig {
                maximum_scene_bytes: 0,
                ..test_config()
            },
            ServerConfig {
                maximum_scene_bytes: MAXIMUM_SCENE_BYTES_LIMIT + 1,
                ..test_config()
            },
            ServerConfig {
                expire_after: Duration::from_millis(100),
                stale_after: Duration::from_millis(100),
                ..test_config()
            },
            ServerConfig {
                websocket_send_deadline: Duration::ZERO,
                ..test_config()
            },
            ServerConfig {
                maximum_websocket_clients: 0,
                ..test_config()
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn maximum_scene_payload_has_a_hard_memory_bound() {
        let mut config = test_config();
        config.maximum_scene_bytes = MAXIMUM_SCENE_BYTES_LIMIT;
        assert!(config.validate().is_ok());
        config.maximum_scene_bytes += 1;
        assert!(config
            .validate()
            .unwrap_err()
            .contains("maximum scene bytes"));
    }

    #[tokio::test]
    async fn no_localization_returns_explicit_not_ready_and_health_liveness_is_independent() {
        let gateway = Gateway::new(test_config()).unwrap();
        let router = gateway.router();
        let (status, body) = get_json(router.clone(), "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"], "localization_unavailable");
        let (ready_status, ready) = get_json(router, "/api/v1/health/ready").await;
        assert_eq!(ready_status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(ready["status"], "not_ready");
        let (live_status, live) = get_json(gateway.router(), "/api/v1/health/live").await;
        assert_eq!(live_status, StatusCode::OK);
        assert_eq!(live["status"], "alive");
    }

    #[tokio::test]
    async fn rust_gateway_serves_session_metadata_and_resets_without_changing_scene_locations() {
        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            stale_after: Duration::from_secs(30),
            expire_after: Duration::from_secs(60),
            ..test_config()
        })
        .unwrap();
        gateway
            .activate_producer_epoch("engine-session-http".into(), "session-http".into())
            .unwrap();
        gateway
            .accept_scene(scene_for(
                "engine-session-http",
                "session-http",
                1,
                "frame-session-http",
            ))
            .unwrap();

        let router = gateway.router();
        let (status, before) = get_json(router.clone(), "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(before["session"]["revision"], 0);
        assert_eq!(before["game_type"], "practice");
        let positions = before["balls"].clone();

        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/game/reset?game_type=8ball")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let reset: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(reset["success"], true);
        assert_eq!(reset["new_state"]["session"]["revision"], 1);
        assert_eq!(reset["new_state"]["game_type"], "8ball");
        assert_eq!(reset["new_state"]["balls"], positions);

        let (status, after) = get_json(gateway.router(), "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(after["session"]["revision"], 1);
        assert_eq!(after["balls"], positions);
    }

    #[test]
    fn configured_game_session_store_uses_the_effective_revisioned_configuration() {
        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let store = gateway
            .inner
            .game_session_store
            .as_ref()
            .expect("configured session store should be available");
        assert_eq!(
            store.load().unwrap(),
            crate::game_session::GameSessionSnapshotV1::initial()
        );
        assert!(!directory.path().join("game-session.json").exists());
        assert!(directory.path().join("game-session.json.lock").exists());
    }

    #[test]
    fn game_session_store_paths_cannot_escape_or_follow_a_symlink_outside_project_root() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("data")).unwrap();
        let valid =
            super::resolve_store_path_under_root(PathBuf::from("data/session.json"), root.path())
                .unwrap();
        assert_eq!(
            valid,
            std::fs::canonicalize(root.path())
                .unwrap()
                .join("data/session.json")
        );
        assert!(super::resolve_store_path_under_root(
            outside.path().join("session.json"),
            root.path(),
        )
        .is_err());
        assert!(super::resolve_store_path_under_root(
            PathBuf::from("../outside/session.json"),
            root.path(),
        )
        .is_err());

        let link = root.path().join("external");
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), &link).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(outside.path(), &link).unwrap();
        assert!(
            super::resolve_store_path_under_root(link.join("session.json"), root.path(),).is_err()
        );
    }

    #[tokio::test]
    async fn http_state_exposes_the_active_projection_and_matching_revision() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-projection".into(), "session-projection".into())
            .unwrap();
        gateway
            .accept_scene(scene_for(
                "engine-projection",
                "session-projection",
                1,
                "frame-projection",
            ))
            .unwrap();
        let calibration: crate::calibration::TableCalibrationV1 = serde_json::from_str(
            include_str!("../contracts/calibration/v1/table-calibration-perspective.json"),
        )
        .unwrap();
        let calibration = calibration.into_validated().unwrap();
        let mut perception = test_perception_state(1, true, false);
        perception.layout_basis_frame_id = Some("frame-projection".into());
        perception.image_width_px = Some(1000);
        perception.image_height_px = Some(500);
        perception.decision_calibration_revision = Some(3);
        perception.decision_rectification_revision = Some(2);
        perception.table_projection = Some(
            crate::capture_runtime::table_projection_from_calibration(&calibration),
        );
        gateway
            .accept_perception_state(1, "engine-projection", "session-projection", perception)
            .unwrap();

        let snapshot = gateway.snapshot_now().unwrap();
        let response = http_scene_response(&snapshot);
        assert_eq!(response["table_calibration_revision"], 3);
        assert_eq!(
            response["table"]["direction"],
            "source_image_px_to_table_mm"
        );
        assert_eq!(response["table"]["table_width_mm"], 2000.0);
        assert_eq!(
            response["table"]["pixel_to_table_mm"][0][0],
            calibration.pixel_to_table_mm()[0][0]
        );
    }

    #[tokio::test]
    async fn accepted_scene_is_available_over_http_and_latest_only_websocket() {
        let gateway = Gateway::new(test_config()).unwrap();
        let engine = "engine-rust-stage2-a";
        let session = "replay-session-01";
        gateway
            .activate_producer_epoch(engine.into(), session.into())
            .unwrap();
        let scene = fixture_scene();
        gateway.accept_scene(scene).unwrap();

        let response = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let (status, body) = get_json(gateway.router(), "/api/v1/game/state").await;
                if status == StatusCode::OK {
                    break body;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(response["perception_source"], "rust_scene");
        assert_eq!(response["frame_number"], 1);
        assert_eq!(response["table"], Value::Null);
        assert_eq!(response["game_type"], Value::Null);
        assert_eq!(response["balls"][0]["ball_type"], "unknown");
        assert_eq!(response["is_valid"], true);
        assert_object_keys(
            &response,
            &[
                "timestamp",
                "frame_number",
                "balls",
                "cue",
                "table",
                "game_type",
                "session",
                "is_valid",
                "confidence",
                "core_metadata_status",
                "core_metadata_timestamp",
                "table_metadata_status",
                "scene_calibration_revision",
                "table_calibration_revision",
                "events",
                "confidence_status",
                "observation_status",
                "perception_source",
            ],
        );
        assert_object_keys(
            &response["balls"][0],
            &[
                "id",
                "number",
                "position",
                "velocity",
                "is_cue_ball",
                "is_pocketed",
                "is_observed",
                "confidence",
                "last_update",
                "velocity_status",
                "cue_ball_status",
                "pocket_status",
                "confidence_kind",
                "last_observation_confidence",
                "ball_type",
                "identity_status",
                "identity_confidence",
                "identity_evidence",
                "tracking_status",
                "position_kind",
                "table_position",
                "source_observation",
                "radius",
                "radius_extent",
                "radius_unit",
                "radius_source",
                "radius_scale",
                "image_space",
                "image_width_px",
                "image_height_px",
            ],
        );
        let legacy_http_contract: Value = serde_json::from_slice(include_bytes!(
            "../contracts/public-state/v1/fixtures/game-state-http.json"
        ))
        .unwrap();
        for key in legacy_http_contract.as_object().unwrap().keys() {
            assert!(
                response.get(key).is_some(),
                "missing HTTP contract field {key}"
            );
        }
        for key in legacy_http_contract["balls"][0].as_object().unwrap().keys() {
            assert!(
                response["balls"][0].get(key).is_some(),
                "missing ball contract field {key}"
            );
        }

        let websocket_contract: Value = serde_json::from_slice(include_bytes!(
            "../contracts/public-state/v1/fixtures/websocket-state-rust-scene.json"
        ))
        .unwrap();
        let http_scene_projection = json!({
            "ball_count": response["balls"].as_array().unwrap().len(),
            "balls": response["balls"].as_array().unwrap().iter().map(|ball| json!({
                "id": ball["id"],
                "position": ball["position"],
                "table_position": ball["table_position"],
                "radius": ball["radius"],
                "radius_extent": ball["radius_extent"],
                "radius_scale": ball["radius_scale"],
                "radius_unit": ball["radius_unit"],
                "radius_source": ball["radius_source"],
                "confidence": ball["confidence"],
                "confidence_kind": ball["confidence_kind"],
                "last_observation_confidence": ball["last_observation_confidence"],
                "number": ball["number"],
                "type": ball["ball_type"],
                "identity_status": ball["identity_status"],
                "identity_confidence": ball["identity_confidence"],
                "identity_evidence": ball["identity_evidence"],
                "is_observed": ball["is_observed"],
                "tracking_status": ball["tracking_status"],
                "position_kind": ball["position_kind"],
                "source_observation": ball["source_observation"],
                "image_space": ball["image_space"],
                "image_width_px": ball["image_width_px"],
                "image_height_px": ball["image_height_px"],
            })).collect::<Vec<_>>(),
            "cue": response["cue"],
            "table": response["table"],
            "game_type": response["game_type"],
            "frame_number": response["frame_number"],
            "observation_status": response["observation_status"],
            "sequence": response["frame_number"],
            "timestamp": response["timestamp"],
        });
        let mut normalized_http_projection = http_scene_projection;
        let mut normalized_contract = websocket_contract["data"].clone();
        // The scene fixture intentionally has no calibration, while the reusable WS
        // contract sample includes a calibrated point.
        normalized_contract["balls"][0]["table_position"] = Value::Null;
        mask_contract_time_and_sequence(&mut normalized_http_projection);
        mask_contract_time_and_sequence(&mut normalized_contract);
        assert_fixture_fields(
            &normalized_http_projection,
            &normalized_contract,
            "HTTP scene projection",
        );

        let http_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_address = http_listener.local_addr().unwrap();
        let app = gateway.router();
        tokio::spawn(async move { axum::serve(http_listener, app).await.unwrap() });
        let (mut client, _) =
            tokio_tungstenite::connect_async(format!("ws://{http_address}/api/v1/websocket/ws"))
                .await
                .unwrap();
        let message = tokio::time::timeout(Duration::from_secs(1), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let ClientMessage::Text(encoded) = message else {
            panic!("expected initial websocket state message")
        };
        let envelope: Value = serde_json::from_str(&encoded).unwrap();
        let contract: Value = serde_json::from_slice(include_bytes!(
            "../contracts/public-state/v1/fixtures/websocket-state-rust-scene.json"
        ))
        .unwrap();
        let mut normalized_envelope = envelope.clone();
        let mut normalized_contract = contract;
        // The shared scene fixture carries no calibrated table point. The websocket
        // contract also covers that optional field, so its expected value is null here.
        normalized_contract["data"]["balls"][0]["table_position"] = Value::Null;
        mask_contract_time_and_sequence(&mut normalized_envelope);
        mask_contract_time_and_sequence(&mut normalized_contract);
        assert_object_keys(
            &normalized_envelope,
            &["type", "timestamp", "sequence", "data"],
        );
        assert_object_keys(
            &normalized_envelope["data"],
            &[
                "ball_count",
                "balls",
                "cue",
                "table",
                "game_type",
                "frame_number",
                "observation_status",
                "sequence",
                "timestamp",
            ],
        );
        assert_object_keys(
            &normalized_envelope["data"]["balls"][0],
            &[
                "id",
                "position",
                "table_position",
                "radius",
                "radius_extent",
                "radius_scale",
                "radius_unit",
                "radius_source",
                "confidence",
                "confidence_kind",
                "last_observation_confidence",
                "number",
                "type",
                "identity_status",
                "identity_confidence",
                "identity_evidence",
                "is_observed",
                "tracking_status",
                "position_kind",
                "source_observation",
                "image_space",
                "image_width_px",
                "image_height_px",
            ],
        );
        assert_fixture_fields(&normalized_envelope, &normalized_contract, "state");

        let next_scene = scene_for(engine, session, 2, "frame-2");
        gateway.accept_scene(next_scene).unwrap();
        let next = tokio::time::timeout(Duration::from_secs(1), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let ClientMessage::Text(next) = next else {
            panic!("expected a newer websocket state")
        };
        let next: Value = serde_json::from_str(&next).unwrap();
        assert_eq!(next["sequence"], 2);
        assert_eq!(
            next["data"]["observation_status"]["source_frame_id"],
            "frame-2"
        );
    }

    #[tokio::test]
    async fn gateway_websocket_audit_is_independent_and_records_selected_state() {
        let audit_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let audit_address = audit_listener.local_addr().unwrap();
        let audit_receiver = std::thread::spawn(move || {
            let (mut stream, _) = audit_listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let send: Value = serde_json::from_slice(&read_audit_record(&mut stream)).unwrap();
            let footer: Value = serde_json::from_slice(&read_audit_record(&mut stream)).unwrap();
            (send, footer)
        });
        let exporter = LiveAuditExporter::start(LiveAuditConfig {
            address: audit_address,
            queue_capacity: 32,
            maximum_record_bytes: 8_192,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
        })
        .unwrap();
        let gateway = Gateway::new_with_audit(
            test_config(),
            Some((exporter.recorder(), "gateway-audit-test".to_owned())),
        )
        .unwrap();
        gateway
            .activate_producer_epoch("audit-engine".into(), "audit-session".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("audit-engine", "audit-session", 1, "audit-frame"))
            .unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = gateway.router();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (mut client, _) =
            tokio_tungstenite::connect_async(format!("ws://{address}/api/v1/websocket/ws"))
                .await
                .unwrap();
        let message = tokio::time::timeout(Duration::from_secs(1), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(message, ClientMessage::Text(_)));
        client.close(None).await.unwrap();
        server.abort();

        let summary = exporter.finish();
        let (send, footer) = audit_receiver.join().unwrap();
        assert_eq!(summary.gateway_websocket_send_successes, 1);
        assert_eq!(send["record_type"], "gateway_websocket_send");
        assert_eq!(send["run_id"], "gateway-audit-test");
        assert_eq!(send["outcome"], "sent");
        assert_eq!(send["publication_sequence"], 1);
        assert_eq!(send["engine_instance_id"], "audit-engine");
        assert_eq!(send["source_session_id"], "audit-session");
        assert!(send["source_frame_id"].is_string());
        assert!(send["socket_send_call_ms"].is_number());
        assert_eq!(send["watch_wakeup_to_consume_ms"], Value::Null);
        assert_eq!(footer["gateway_websocket_send_successes"], 1);
    }

    #[test]
    fn gateway_websocket_audit_counts_skipped_publications_and_source_frames() {
        let audit_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let audit_address = audit_listener.local_addr().unwrap();
        let audit_receiver = std::thread::spawn(move || {
            let (mut stream, _) = audit_listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let first: Value = serde_json::from_slice(&read_audit_record(&mut stream)).unwrap();
            let second: Value = serde_json::from_slice(&read_audit_record(&mut stream)).unwrap();
            let footer: Value = serde_json::from_slice(&read_audit_record(&mut stream)).unwrap();
            (first, second, footer)
        });
        let exporter = LiveAuditExporter::start(LiveAuditConfig {
            address: audit_address,
            queue_capacity: 32,
            maximum_record_bytes: 8_192,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
        })
        .unwrap();
        let gateway = Gateway::new_with_audit(
            test_config(),
            Some((exporter.recorder(), "gateway-gap-test".to_owned())),
        )
        .unwrap();
        assert!(Gateway::new(test_config())
            .unwrap()
            .inner
            .gateway_audit
            .is_none());
        gateway
            .activate_producer_epoch("gap-engine".into(), "gap-session".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("gap-engine", "gap-session", 1, "frame-1"))
            .unwrap();
        let first = gateway.inner.latest.borrow().as_ref().unwrap().clone();
        let audit = Arc::clone(gateway.inner.gateway_audit.as_ref().unwrap());
        let mut client = GatewayWebSocketAuditClient {
            audit,
            client_id: 1,
            previous_publication_sequence: None,
            previous_source_identity: None,
        };
        client.record_send(&first, 0.5, None, 0.1, 0.2, Some(0.3), Some(256), "sent");

        gateway
            .accept_scene(scene_for("gap-engine", "gap-session", 2, "frame-2"))
            .unwrap();
        gateway
            .accept_scene(scene_for("gap-engine", "gap-session", 4, "frame-4"))
            .unwrap();
        let latest = gateway.inner.latest.borrow().as_ref().unwrap().clone();
        client.record_send(
            &latest,
            0.5,
            Some(0.01),
            0.1,
            0.2,
            Some(0.3),
            Some(256),
            "sent",
        );

        exporter.finish();
        let (first, second, footer) = audit_receiver.join().unwrap();
        assert_eq!(first["publication_sequence_gap"], 0);
        assert_eq!(first["source_sequence_gap"], Value::Null);
        assert_eq!(second["publication_sequence"], 3);
        assert_eq!(second["publication_sequence_gap"], 1);
        assert_eq!(second["source_sequence_gap"], 2);
        assert_eq!(second["source_frame_id"], "frame-4");
        assert_eq!(second["repeated_publication"], false);
        assert_eq!(second["repeated_source_observation"], false);
        assert_eq!(footer["gateway_websocket_send_successes"], 2);
    }

    fn read_audit_record(stream: &mut std::net::TcpStream) -> Vec<u8> {
        use std::io::Read;
        let mut header = [0_u8; 4];
        stream.read_exact(&mut header).unwrap();
        let mut payload = vec![0; u32::from_be_bytes(header) as usize];
        stream.read_exact(&mut payload).unwrap();
        payload
    }

    #[tokio::test]
    async fn websocket_upgrade_uses_newest_state_if_scene_changes_during_handshake() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 1, "frame-1"))
            .unwrap();

        // This is the same ordering as websocket_upgrade + websocket_session: subscribe
        // before the handshake, then read with borrow_and_update after it completes.
        let mut latest = gateway.inner.latest.subscribe();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 2, "frame-2"))
            .unwrap();
        let initial = take_latest_state(&mut latest).unwrap();
        assert_eq!(
            initial.current.payload.observation_status["source_frame_id"],
            "frame-2"
        );
        assert_eq!(initial.publication_sequence, 2);
    }

    #[tokio::test]
    async fn websocket_phase_aware_producer_state_bypasses_optional_position_retention() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-retain".into(), "session-retain".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-retain", "session-retain", 1, "frame-1"))
            .unwrap();
        gateway
            .accept_perception_state(
                1,
                "engine-retain",
                "session-retain",
                test_perception_state(1, true, false),
            )
            .unwrap();

        let (server, mut client) = start_websocket(&gateway, "?retain_last_known_ms=500").await;
        let initial = receive_websocket_state(&mut client).await;
        let original_ball_count = initial["data"]["ball_count"].as_u64().unwrap();
        assert!(original_ball_count > 0);
        assert!(initial["data"]["observation_status"]["position_retention"].is_null());
        assert_eq!(
            initial["data"]["observation_status"]["perception"]["layout_trusted"], true,
            "producer trust is preserved while gateway retention is bypassed"
        );
        assert_eq!(
            initial["data"]["balls"],
            websocket_scene_message(&gateway.snapshot_now().unwrap())["data"]["balls"]
        );

        gateway
            .accept_perception_state(
                2,
                "engine-retain",
                "session-retain",
                test_perception_state(2, false, false),
            )
            .unwrap();
        let stale = receive_websocket_state(&mut client).await;
        assert_eq!(stale["data"]["ball_count"], original_ball_count);
        assert_eq!(
            stale["data"]["balls"],
            websocket_scene_message(&gateway.snapshot_now().unwrap())["data"]["balls"],
            "stale producer positions remain exactly producer-owned"
        );
        assert!(stale["data"]["observation_status"]["position_retention"].is_null());

        let mut empty_scene = scene_for(
            "engine-retain",
            "session-retain",
            2,
            "frame-empty-unverified",
        );
        empty_scene.balls.clear();
        gateway.accept_scene(empty_scene).unwrap();
        let retained = receive_websocket_state(&mut client).await;
        assert_eq!(retained["data"]["ball_count"], 0);
        assert_eq!(
            retained["data"]["balls"],
            websocket_scene_message(&gateway.snapshot_now().unwrap())["data"]["balls"],
            "a clean empty replacement cannot be repopulated from gateway cache"
        );
        assert_eq!(
            retained["data"]["observation_status"]["ball_positions_status"], "unavailable",
            "phase-aware position status is passed through exactly from producer"
        );
        assert_eq!(
            retained["data"]["observation_status"]["perception"]["layout_trusted"], false,
            "gateway output retains the producer's stale-state decision"
        );
        assert_eq!(
            retained["data"]["observation_status"]["perception"]["guidance_valid"],
            false
        );
        assert!(retained["data"]["observation_status"]["position_retention"].is_null());

        gateway
            .accept_scene(scene_for(
                "engine-retain",
                "session-retain",
                3,
                "frame-recovered",
            ))
            .unwrap();
        let recovered = receive_websocket_state(&mut client).await;
        assert!(recovered["data"]["ball_count"].as_u64().unwrap() > 0);
        assert!(recovered["data"]["observation_status"]["position_retention"].is_null());

        let mut empty_again = scene_for("engine-retain", "session-retain", 4, "frame-empty-again");
        empty_again.balls.clear();
        gateway.accept_scene(empty_again).unwrap();
        let retained_again = receive_websocket_state(&mut client).await;
        assert_eq!(retained_again["data"]["ball_count"], 0);
        assert!(retained_again["data"]["observation_status"]["position_retention"].is_null());

        let mut verified_empty_state = test_perception_state(3, true, true);
        verified_empty_state.layout_basis_frame_id = Some("frame-empty-again".into());
        gateway
            .accept_perception_state(3, "engine-retain", "session-retain", verified_empty_state)
            .unwrap();
        let verified_empty = receive_websocket_state(&mut client).await;
        assert_eq!(verified_empty["data"]["ball_count"], 0);
        assert_eq!(
            verified_empty["data"]["observation_status"]["perception"]["empty_layout_verified"],
            true,
            "verified empty state was rejected: {:?}",
            verified_empty["data"]["observation_status"]
        );
        assert!(verified_empty["data"]["observation_status"]["position_retention"].is_null());

        gateway
            .accept_perception_state(
                4,
                "engine-retain",
                "session-retain",
                test_perception_state(4, false, false),
            )
            .unwrap();
        let after_verified_empty = receive_websocket_state(&mut client).await;
        assert_eq!(after_verified_empty["data"]["ball_count"], 0);
        assert!(after_verified_empty["data"]["observation_status"]["position_retention"].is_null());

        client.close(None).await.unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn websocket_position_retention_expires_and_sends_the_expiry_state() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-expire".into(), "session-expire".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-expire", "session-expire", 1, "frame-1"))
            .unwrap();
        let (server, mut client) = start_websocket(&gateway, "?retain_last_known_ms=60").await;
        let initial = receive_websocket_state(&mut client).await;
        let original_ball_count = initial["data"]["ball_count"].as_u64().unwrap();

        // The grace period is an unavailable-state debounce, not a maximum age
        // measured from an otherwise healthy layout.
        tokio::time::sleep(Duration::from_millis(80)).await;
        let mut empty_scene = scene_for("engine-expire", "session-expire", 2, "frame-empty");
        empty_scene.balls.clear();
        gateway.accept_scene(empty_scene).unwrap();
        let retained = receive_websocket_state(&mut client).await;
        assert_eq!(retained["data"]["ball_count"], original_ball_count);
        assert_eq!(
            retained["data"]["observation_status"]["position_retention"]["retained"],
            true
        );
        assert!(
            retained["data"]["observation_status"]["position_retention"]["retained_age_ms"]
                .as_u64()
                .unwrap()
                >= 60
        );
        assert!(
            retained["data"]["observation_status"]["position_retention"]["unavailable_age_ms"]
                .as_u64()
                .unwrap()
                < 60
        );

        let expired = receive_websocket_state(&mut client).await;
        assert_eq!(expired["data"]["ball_count"], 0);
        assert_eq!(
            expired["data"]["observation_status"]["position_retention"]["retained"],
            false
        );
        assert_eq!(
            expired["data"]["observation_status"]["position_retention"]["grace_ms"],
            60
        );

        client.close(None).await.unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn websocket_position_retention_resets_when_source_or_calibration_changes() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-context".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-context", "session-a", 1, "frame-1"))
            .unwrap();
        let (server, mut client) = start_websocket(&gateway, "?retain_last_known_ms=500").await;
        let initial = receive_websocket_state(&mut client).await;
        assert!(initial["data"]["ball_count"].as_u64().unwrap() > 0);

        let mut changed_calibration = scene_for(
            "engine-context",
            "session-a",
            2,
            "frame-calibration-changed",
        );
        changed_calibration.revisions.calibration = Some(
            changed_calibration
                .revisions
                .calibration
                .unwrap_or_default()
                .saturating_add(1),
        );
        changed_calibration.revisions.calibration_state = crate::CalibrationState::Valid;
        let changed_calibration_revision = changed_calibration.revisions.calibration.unwrap();
        changed_calibration.balls.clear();
        gateway.accept_scene(changed_calibration).unwrap();
        let calibration_changed = receive_websocket_state(&mut client).await;
        assert_eq!(calibration_changed["data"]["ball_count"], 0);
        assert_eq!(
            calibration_changed["data"]["observation_status"]["position_retention"]["retained"],
            false
        );

        let mut observed_after_calibration_change = scene_for(
            "engine-context",
            "session-a",
            3,
            "frame-after-calibration-change",
        );
        observed_after_calibration_change.revisions.calibration =
            Some(changed_calibration_revision);
        observed_after_calibration_change
            .revisions
            .calibration_state = crate::CalibrationState::Valid;
        gateway
            .accept_scene(observed_after_calibration_change)
            .unwrap();
        let reseeded = receive_websocket_state(&mut client).await;
        assert!(reseeded["data"]["ball_count"].as_u64().unwrap() > 0);

        let mut invalid_calibration_state = scene_for(
            "engine-context",
            "session-a",
            4,
            "frame-calibration-invalid",
        );
        invalid_calibration_state.revisions.calibration = Some(changed_calibration_revision);
        invalid_calibration_state.revisions.calibration_state = crate::CalibrationState::Invalid;
        invalid_calibration_state.balls.clear();
        gateway.accept_scene(invalid_calibration_state).unwrap();
        let calibration_state_changed = receive_websocket_state(&mut client).await;
        assert_eq!(calibration_state_changed["data"]["ball_count"], 0);
        assert_eq!(
            calibration_state_changed["data"]["observation_status"]["position_retention"]
                ["retained"],
            false,
            "calibration validity changing without a new revision must clear the cache"
        );

        let mut observed_after_calibration_state_change = scene_for(
            "engine-context",
            "session-a",
            5,
            "frame-after-calibration-state-change",
        );
        observed_after_calibration_state_change
            .revisions
            .calibration = Some(changed_calibration_revision);
        observed_after_calibration_state_change
            .revisions
            .calibration_state = crate::CalibrationState::Valid;
        gateway
            .accept_scene(observed_after_calibration_state_change)
            .unwrap();
        let reseeded_after_calibration_state = receive_websocket_state(&mut client).await;
        assert!(
            reseeded_after_calibration_state["data"]["ball_count"]
                .as_u64()
                .unwrap()
                > 0
        );

        gateway
            .activate_producer_epoch("engine-context".into(), "session-b".into())
            .unwrap();
        let mut new_source_scene = scene_for("engine-context", "session-b", 1, "frame-new-source");
        new_source_scene.revisions.calibration = Some(changed_calibration_revision);
        new_source_scene.revisions.calibration_state = crate::CalibrationState::Valid;
        new_source_scene.balls.clear();
        gateway.accept_scene(new_source_scene).unwrap();
        let source_changed = receive_websocket_state(&mut client).await;
        assert_eq!(source_changed["data"]["ball_count"], 0);
        assert_eq!(
            source_changed["data"]["observation_status"]["position_retention"]["retained"],
            false
        );
        assert_eq!(
            source_changed["data"]["observation_status"]["source_session_id"],
            "session-b"
        );
        assert_eq!(
            source_changed["data"]["observation_status"]["calibration_revision"],
            changed_calibration_revision
        );

        client.close(None).await.unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn websocket_position_retention_is_disabled_without_query_parameter() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-default".into(), "session-default".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-default", "session-default", 1, "frame-1"))
            .unwrap();
        let (server, mut client) = start_websocket(&gateway, "").await;
        let initial = receive_websocket_state(&mut client).await;
        assert!(initial["data"]["ball_count"].as_u64().unwrap() > 0);
        assert!(initial["data"]["observation_status"]["position_retention"].is_null());

        let mut empty_scene = scene_for("engine-default", "session-default", 2, "frame-empty");
        empty_scene.balls.clear();
        gateway.accept_scene(empty_scene).unwrap();
        let empty = receive_websocket_state(&mut client).await;
        assert_eq!(empty["data"]["ball_count"], 0);
        assert!(empty["data"]["observation_status"]["position_retention"].is_null());

        client.close(None).await.unwrap();
        server.abort();

        gateway
            .accept_scene(scene_for(
                "engine-default",
                "session-default",
                3,
                "frame-restored",
            ))
            .unwrap();
        let (zero_server, mut zero_client) =
            start_websocket(&gateway, "?retain_last_known_ms=0").await;
        let zero_initial = receive_websocket_state(&mut zero_client).await;
        assert!(zero_initial["data"]["ball_count"].as_u64().unwrap() > 0);
        assert!(zero_initial["data"]["observation_status"]["position_retention"].is_null());

        let mut zero_empty_scene =
            scene_for("engine-default", "session-default", 4, "frame-zero-empty");
        zero_empty_scene.balls.clear();
        gateway.accept_scene(zero_empty_scene).unwrap();
        let zero_empty = receive_websocket_state(&mut zero_client).await;
        assert_eq!(zero_empty["data"]["ball_count"], 0);
        assert!(zero_empty["data"]["observation_status"]["position_retention"].is_null());

        zero_client.close(None).await.unwrap();
        zero_server.abort();
    }

    #[tokio::test]
    async fn concurrent_snapshot_reads_cannot_overwrite_the_latest_ingest() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 1, "frame-1"))
            .unwrap();
        let mut latest = gateway.inner.latest.subscribe();
        let reader_gateway = gateway.clone();
        let start = Arc::new(std::sync::Barrier::new(2));
        let reader_start = Arc::clone(&start);
        let unavailable_snapshot = Arc::new(AtomicBool::new(false));
        let reader_unavailable = Arc::clone(&unavailable_snapshot);
        let reader = tokio::task::spawn_blocking(move || {
            reader_start.wait();
            for _ in 0..2_000 {
                if reader_gateway.snapshot_now().is_none() {
                    reader_unavailable.store(true, Ordering::Release);
                }
                std::thread::yield_now();
            }
        });
        start.wait();

        for sequence in 2..=64 {
            gateway
                .accept_scene(scene_for(
                    "engine-a",
                    "session-a",
                    sequence,
                    &format!("frame-{sequence}"),
                ))
                .unwrap();
            tokio::task::yield_now().await;
        }
        reader.await.unwrap();
        assert!(!unavailable_snapshot.load(Ordering::Acquire));
        let newest = gateway.snapshot_now().unwrap();
        let watched = take_latest_state(&mut latest).unwrap();
        assert_eq!(
            watched.current.payload.observation_status["source_frame_id"],
            newest.current.payload.observation_status["source_frame_id"]
        );
        assert_eq!(watched.publication_sequence, newest.publication_sequence);
    }

    #[test]
    fn scene_validity_matches_the_public_state_contract() {
        let mut status = json!({
            "phase":"rust_scene_observation",
            "ball_detection_status":"observed",
            "ball_positions_status":"observed",
            "freshness_status":"stale",
        });
        assert!(scene_state_is_valid(&status));
        status["ball_positions_status"] = json!("cached");
        assert!(!scene_state_is_valid(&status));
    }

    #[test]
    fn failed_live_ready_write_does_not_commit_a_new_producer_epoch() {
        struct FailedWriter;
        impl Write for FailedWriter {
            fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "simulated ready write failure",
                ))
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let settings = gateway.resolve_live_capture_settings().unwrap();
        let hello = json!({
            "type": "hello",
            "protocol_version": LIVE_PROTOCOL_VERSION,
            "engine_instance_id": "engine-b",
            "source_session_id": "session-b",
            "capture_session_ack": live_acknowledgement(&settings, "session-b"),
        });
        let handshake =
            validate_live_v3_hello(&serde_json::to_vec(&hello).unwrap(), &settings).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 1, "frame-1"))
            .unwrap();

        assert!(gateway
            .acknowledge_live_producer(&mut FailedWriter, &handshake)
            .is_err());
        let retained = gateway.snapshot_now().unwrap();
        assert_eq!(
            retained.current.payload.observation_status["engine_instance_id"],
            "engine-a"
        );

        let mut ready = Vec::new();
        gateway
            .acknowledge_live_producer(&mut ready, &handshake)
            .unwrap();
        assert!(gateway.snapshot_now().is_none());
        let length = u32::from_be_bytes(ready[..4].try_into().unwrap()) as usize;
        let response: Value = serde_json::from_slice(&ready[4..4 + length]).unwrap();
        assert_eq!(response["protocol_version"], LIVE_PROTOCOL_VERSION);
        assert_eq!(response["last_transport_sequence"], Value::Null);
    }

    #[tokio::test]
    async fn http_ball_age_uses_the_same_read_instant_as_monotonic_age_for_reused_balls() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        let first = scene_for("engine-a", "session-a", 1, "frame-1");
        gateway.accept_scene(first.clone()).unwrap();
        let first_received_at = lock_unpoisoned(&gateway.inner.state)
            .localization_received_at
            .unwrap();
        tokio::time::sleep(Duration::from_millis(60)).await;

        let mut reused = scene_for("engine-a", "session-a", 2, "frame-2");
        let measurement = reused.balls[0].measurement.take().unwrap();
        reused.balls[0].status = crate::BallStatus::Tracked;
        reused.balls[0].last_observation = first.balls[0].last_observation.clone();
        reused.balls[0].estimate = Some(crate::Estimate {
            kind: crate::EstimateKind::Filtered,
            as_of: reused.source.timestamp.clone(),
            image_center_px: Some(measurement.image_center_px),
            image_space: Some(measurement.image_space),
            image_width_px: Some(measurement.image_width_px),
            image_height_px: Some(measurement.image_height_px),
            table_center_mm: None,
            table_position_uncertainty: measurement.table_position_uncertainty,
        });
        reused.balls[0].measurement = None;
        reused.validate().unwrap();
        gateway.accept_scene(reused).unwrap();

        let state = gateway.snapshot_now().unwrap();
        let age_ns = state.current.ball_age_ns["track-1"];
        assert!(age_ns >= 50_000_000);
        let response = http_scene_response(&state);
        let last_update =
            DateTime::parse_from_rfc3339(response["balls"][0]["last_update"].as_str().unwrap())
                .unwrap()
                .with_timezone(&Utc);
        assert!((last_update - first_received_at).num_milliseconds().abs() < 30);
    }

    #[tokio::test]
    async fn accepted_health_advances_state_timestamp_without_refreshing_ball_observation_time() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 1, "frame-1"))
            .unwrap();
        let initial = gateway.snapshot_now().unwrap();
        let initial_http = http_scene_response(&initial);
        let initial_ws = websocket_scene_message(&initial);
        let initial_timestamp =
            DateTime::parse_from_rfc3339(initial_http["timestamp"].as_str().unwrap())
                .unwrap()
                .with_timezone(&Utc);
        let initial_ball_update =
            DateTime::parse_from_rfc3339(initial_http["balls"][0]["last_update"].as_str().unwrap())
                .unwrap()
                .with_timezone(&Utc);
        let initial_localization_receipt =
            initial.current.localization_received_monotonic_ns.unwrap();
        let mut latest = gateway.inner.latest.subscribe();

        tokio::time::sleep(Duration::from_millis(40)).await;
        let mut health: SceneSnapshotV1 =
            serde_json::from_slice(include_bytes!("../contracts/v1/scene-failure.json")).unwrap();
        health.engine_instance_id = "engine-a".into();
        health.source.session_id = "session-a".into();
        health.source.frame_id = "health-frame-2".into();
        health.source.timestamp.ticks = "1000000002".into();
        health.transport_sequence = 2;
        health.snapshot_revision = 2;
        health.validate().unwrap();
        gateway.accept_scene(health).unwrap();

        // The live update and the subsequent HTTP/freshness snapshot must agree on
        // accepted-event time. Localization receipt remains separate for ball ages.
        let health_published = take_latest_state(&mut latest).unwrap();
        let health_ws = websocket_scene_message(&health_published);
        let refreshed = gateway.snapshot_now().unwrap();
        let refreshed_http = http_scene_response(&refreshed);
        let refreshed_ws = websocket_scene_message(&refreshed);
        let refreshed_timestamp =
            DateTime::parse_from_rfc3339(refreshed_http["timestamp"].as_str().unwrap())
                .unwrap()
                .with_timezone(&Utc);
        let refreshed_ball_update = DateTime::parse_from_rfc3339(
            refreshed_http["balls"][0]["last_update"].as_str().unwrap(),
        )
        .unwrap()
        .with_timezone(&Utc);

        assert!(refreshed_timestamp > initial_timestamp);
        assert_eq!(
            health_ws["timestamp"], refreshed_ws["timestamp"],
            "a freshness read must not regress the accepted state timestamp"
        );
        assert_eq!(refreshed_http["timestamp"], refreshed_ws["timestamp"]);
        assert_eq!(
            refreshed.current.localization_received_monotonic_ns,
            Some(initial_localization_receipt),
            "health events must not rejuvenate localization"
        );
        assert!(
            (refreshed_ball_update - initial_ball_update)
                .num_milliseconds()
                .abs()
                <= 10,
            "a health event must not refresh per-ball last_update"
        );
        assert!(
            (refreshed_timestamp - refreshed_ball_update).num_milliseconds() >= 30,
            "state update time should advance independently of ball observation time"
        );
        assert_eq!(
            initial_ws["data"]["balls"][0]["source_observation"],
            refreshed_ws["data"]["balls"][0]["source_observation"]
        );
    }

    #[tokio::test]
    async fn live_v3_requires_current_config_and_preserves_compatible_localization() {
        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let settings = gateway.resolve_live_capture_settings().unwrap();
        let address = start_ingress(gateway.clone()).await;
        let mut stream = TcpStream::connect(address).await.unwrap();
        let engine = "engine-live-test";
        let session = "capture-session-live-test";
        send_frame(
            &mut stream,
            &serde_json::to_vec(&json!({
                "type": "hello",
                "protocol_version": LIVE_PROTOCOL_VERSION,
                "engine_instance_id": engine,
                "source_session_id": session,
                "capture_session_ack": live_acknowledgement(&settings, session),
            }))
            .unwrap(),
        )
        .await;
        let ready: Value = serde_json::from_slice(&read_frame(&mut stream).await).unwrap();
        assert_eq!(ready["type"], "ready");
        assert_eq!(ready["protocol_version"], LIVE_PROTOCOL_VERSION);
        assert_eq!(
            ready["configuration_revision"],
            settings.configuration_revision
        );
        assert_eq!(ready["request_sha256"], settings.request_sha256);
        assert!(ready["last_transport_sequence"].is_null());
        assert_object_keys(
            &ready,
            &[
                "type",
                "protocol_version",
                "configuration_revision",
                "request_sha256",
                "last_transport_sequence",
            ],
        );

        let mut snapshot = scene_for(engine, session, 1, "live-frame-1");
        let geometry = settings.expected.output_frame_geometry().unwrap();
        snapshot.source.width_px = geometry.width_px();
        snapshot.source.height_px = geometry.height_px();
        for ball in &mut snapshot.balls {
            if let Some(measurement) = ball.measurement.as_mut() {
                measurement.image_width_px = geometry.width_px();
                measurement.image_height_px = geometry.height_px();
            }
            if let Some(estimate) = ball.estimate.as_mut() {
                estimate.image_width_px = Some(geometry.width_px());
                estimate.image_height_px = Some(geometry.height_px());
            }
        }
        let mut producer_lifecycle = SceneStateLifecycle::default();
        producer_lifecycle.activate_source(engine, session).unwrap();
        producer_lifecycle
            .ingest(&snapshot, 100, 100, None, None)
            .unwrap();
        let producer_state = producer_lifecycle.current(100).unwrap();
        let delivery = crate::public_state_lifecycle::PublicStateDeliveryV1::new(
            &snapshot,
            producer_state.as_ref(),
        );
        send_frame(&mut stream, &serde_json::to_vec(&delivery).unwrap()).await;

        let mut state_response = None;
        for _ in 0..30 {
            let (status, response) = get_json(gateway.router(), "/api/v1/game/state").await;
            if status == StatusCode::OK {
                state_response = Some(response);
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let state = state_response.expect("Rust gateway should publish live localization");
        assert_eq!(state["frame_number"], 1);
        assert_eq!(
            state["balls"].as_array().unwrap().len(),
            snapshot.balls.len()
        );
        assert_eq!(state["observation_status"]["source_session_id"], session);
        assert_eq!(state["observation_status"]["freshness_status"], "fresh");
        stream.shutdown().await.unwrap();
    }

    #[test]
    fn live_delivery_rejects_unsupported_versions_and_wrong_source_identity() {
        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let settings = gateway.resolve_live_capture_settings().unwrap();
        let engine = "engine-live-contract";
        let session = "session-live-contract";
        let hello = json!({
            "type": "hello",
            "protocol_version": LIVE_PROTOCOL_VERSION,
            "engine_instance_id": engine,
            "source_session_id": session,
            "capture_session_ack": live_acknowledgement(&settings, session),
        });
        let handshake =
            validate_live_v3_hello(&serde_json::to_vec(&hello).unwrap(), &settings).unwrap();
        let scene = scene_for(engine, session, 1, "frame-live-contract");
        let mut delivery: Value = serde_json::from_slice(&public_state_delivery(&scene)).unwrap();

        delivery["delivery_schema_version"] = json!(2);
        let error = gateway
            .accept_live_delivery(&serde_json::to_vec(&delivery).unwrap(), &handshake)
            .unwrap_err();
        assert!(error.contains("unsupported public-state delivery contract"));

        delivery["delivery_schema_version"] = json!(1);
        delivery["source_snapshot"]["engine_instance_id"] = json!("another-engine");
        let error = gateway
            .accept_live_delivery(&serde_json::to_vec(&delivery).unwrap(), &handshake)
            .unwrap_err();
        assert!(error.contains("scene identity differs from the acknowledged live source"));
    }

    #[test]
    fn v2_scene_and_partial_perception_are_published_as_one_gateway_update() {
        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let settings = gateway.resolve_live_capture_settings().unwrap();
        let engine = "engine-atomic-scene-perception";
        let session = "session-atomic-scene-perception";
        let hello = json!({
            "type": "hello",
            "protocol_version": LIVE_PROTOCOL_VERSION,
            "engine_instance_id": engine,
            "source_session_id": session,
            "capture_session_ack": live_acknowledgement(&settings, session),
        });
        let handshake =
            validate_live_v3_hello(&serde_json::to_vec(&hello).unwrap(), &settings).unwrap();
        gateway
            .activate_producer_epoch(engine.to_owned(), session.to_owned())
            .unwrap();

        gateway
            .accept_live_delivery(
                &public_state_delivery_v2(
                    &scene_for(engine, session, 1, "frame-1"),
                    1,
                    test_guidance_state(1, true),
                ),
                &handshake,
            )
            .unwrap();
        let trusted = gateway
            .inner
            .latest
            .borrow()
            .as_ref()
            .expect("the first live scene should be published")
            .clone();
        assert_eq!(
            trusted.current.payload.observation_status["perception"]["guidance_valid"],
            true
        );
        let publication_before_partial = trusted.publication_sequence;

        gateway
            .accept_live_delivery(
                &public_state_delivery_v2(
                    &scene_for(engine, session, 2, "frame-2"),
                    2,
                    test_perception_state(2, false, false),
                ),
                &handshake,
            )
            .unwrap();

        let partial = gateway
            .inner
            .latest
            .borrow()
            .as_ref()
            .expect("the partial scene should be published")
            .clone();
        assert_eq!(
            partial.publication_sequence,
            publication_before_partial + 1,
            "the scene and perception envelope must cause a single watch publication"
        );
        assert_eq!(
            partial.current.payload.observation_status["source_frame_id"],
            "frame-2"
        );
        assert_eq!(
            partial.current.payload.observation_status["perception"]["layout_trusted"],
            false
        );
        assert_eq!(
            partial.current.payload.observation_status["perception"]["guidance_valid"],
            false
        );
        assert_eq!(partial.current.payload.balls.len(), 1);
    }

    #[test]
    fn gateway_trust_restoration_requires_the_clean_layout_scene_to_be_accepted() {
        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let settings = gateway.resolve_live_capture_settings().unwrap();
        let engine = "engine-layout-basis";
        let session = "session-layout-basis";
        let hello = json!({
            "type": "hello",
            "protocol_version": LIVE_PROTOCOL_VERSION,
            "engine_instance_id": engine,
            "source_session_id": session,
            "capture_session_ack": live_acknowledgement(&settings, session),
        });
        let handshake =
            validate_live_v3_hello(&serde_json::to_vec(&hello).unwrap(), &settings).unwrap();
        gateway
            .activate_producer_epoch(engine.to_owned(), session.to_owned())
            .unwrap();

        gateway
            .accept_live_delivery(
                &public_state_delivery_v2(
                    &scene_for(engine, session, 1, "frame-1"),
                    1,
                    test_perception_state(1, false, false),
                ),
                &handshake,
            )
            .unwrap();

        let mut restored = test_perception_state(2, true, false);
        restored.layout_revision = 2;
        restored.layout_basis_frame_id = Some("frame-2".to_owned());
        gateway
            .accept_perception_state(2, engine, session, restored.clone())
            .unwrap();
        let dropped_clean_scene = gateway
            .inner
            .latest
            .borrow()
            .as_ref()
            .expect("the fail-closed phase event should remain publishable")
            .clone();
        assert_eq!(
            dropped_clean_scene.current.payload.observation_status["source_frame_id"],
            "frame-1"
        );
        assert_eq!(
            dropped_clean_scene.current.payload.observation_status["perception"]["layout_trusted"],
            false,
            "phase-only trust cannot be restored before its clean scene arrives"
        );
        assert_eq!(
            dropped_clean_scene.current.payload.observation_status["perception"]["guidance_valid"],
            false
        );
        assert_eq!(
            dropped_clean_scene.current.payload.balls.len(),
            scene_for(engine, session, 1, "frame-1").balls.len(),
            "fail-closed trust handling preserves the gateway's last-known centers"
        );

        let mut delayed_partial = restored.clone();
        delayed_partial.event_sequence = 3;
        gateway
            .accept_live_delivery(
                &public_state_delivery_v2(
                    &scene_for(engine, session, 3, "frame-3"),
                    3,
                    delayed_partial,
                ),
                &handshake,
            )
            .unwrap();
        assert_eq!(
            gateway
                .inner
                .latest
                .borrow()
                .as_ref()
                .unwrap()
                .current
                .payload
                .observation_status["perception"]["layout_trusted"],
            false,
            "a later scene carrying the stale basis must not restore trust either"
        );

        restored.event_sequence = 4;
        restored.layout_revision = 3;
        restored.layout_basis_frame_id = Some("frame-4".to_owned());
        gateway
            .accept_live_delivery(
                &public_state_delivery_v2(&scene_for(engine, session, 4, "frame-4"), 4, restored),
                &handshake,
            )
            .unwrap();
        assert_eq!(
            gateway
                .inner
                .latest
                .borrow()
                .as_ref()
                .unwrap()
                .current
                .payload
                .observation_status["perception"]["layout_trusted"],
            true,
            "a clean accepted scene can establish its matching layout basis"
        );
    }

    #[test]
    fn perception_events_do_not_renew_ball_age_or_consume_scene_sequence_space() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 1, "frame-1"))
            .unwrap();
        let initial = gateway.snapshot_now().unwrap();
        let initial_localization_time = initial.current.localization_received_monotonic_ns;
        let initial_ball_observation = initial.current.payload.balls[0].source_observation.clone();
        let initial_ball_age = initial.current.ball_age_ns["track-1"];

        gateway
            .accept_perception_state(
                2,
                "engine-a",
                "session-a",
                PerceptionStateV1 {
                    event_sequence: 2,
                    image_width_px: Some(1920),
                    image_height_px: Some(1080),
                    phase: "shot_underway".into(),
                    layout_revision: 1,
                    layout_trusted: false,
                    layout_basis_frame_id: None,
                    empty_layout_verified: false,
                    guidance_valid: false,
                    trust_reason: "motion_detected".into(),
                    last_ball_observation_age_ms: Some(100),
                    latest_motion_check_age_ms: Some(0),
                    cue_evidence_age_ms: None,
                    motion_evidence_max_age_ms: 200,
                    cue_evidence_max_age_ms: 250,
                    layout_verification_max_age_ms: 1_500,
                    cue: None,
                    decision_frame_id: None,
                    decision_calibration_revision: None,
                    decision_rectification_revision: None,
                    table_visibility: None,
                    table_projection: None,
                },
            )
            .unwrap();

        let after_perception = gateway.snapshot_now().unwrap();
        assert_eq!(
            after_perception.current.localization_received_monotonic_ns, initial_localization_time,
            "phase-only delivery must not renew localization receipt"
        );
        assert_eq!(
            after_perception.current.payload.balls[0].source_observation, initial_ball_observation,
            "phase-only delivery must not replace a ball's source observation"
        );
        assert!(after_perception.current.ball_age_ns["track-1"] >= initial_ball_age);
        assert_eq!(
            after_perception.current.payload.observation_status["perception"]["phase"],
            "shot_underway"
        );
        {
            let state = lock_unpoisoned(&gateway.inner.state);
            assert_eq!(state.last_transport_sequence, Some(1));
            assert_eq!(state.last_delivery_sequence, Some(2));
        }

        gateway
            .accept_scene(scene_for("engine-a", "session-a", 2, "frame-2"))
            .expect("a phase event must not make the next raw scene look stale");
        let state = lock_unpoisoned(&gateway.inner.state);
        assert_eq!(state.last_transport_sequence, Some(2));
        assert_eq!(state.last_delivery_sequence, Some(2));
    }

    #[tokio::test]
    async fn phase_and_cue_state_is_available_before_first_ball_localization() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-cue-only".into(), "session-cue-only".into())
            .unwrap();
        let mut latest = gateway.inner.latest.subscribe();

        gateway
            .accept_perception_state(
                1,
                "engine-cue-only",
                "session-cue-only",
                PerceptionStateV1 {
                    event_sequence: 1,
                    image_width_px: Some(1280),
                    image_height_px: Some(720),
                    phase: "pre_shot".into(),
                    layout_revision: 0,
                    layout_trusted: false,
                    layout_basis_frame_id: None,
                    empty_layout_verified: false,
                    guidance_valid: false,
                    trust_reason: "no_verified_layout".into(),
                    last_ball_observation_age_ms: None,
                    latest_motion_check_age_ms: Some(0),
                    cue_evidence_age_ms: Some(0),
                    motion_evidence_max_age_ms: 200,
                    cue_evidence_max_age_ms: 250,
                    layout_verification_max_age_ms: 1_500,
                    cue: Some(crate::public_state_lifecycle::CueGeometryStateV1 {
                        shaft_start_px: [100.0, 200.0],
                        shaft_end_px: [500.0, 180.0],
                        axis_unit: [0.998_752, -0.049_938],
                        confidence: 0.9,
                        observation_age_ms: 0,
                        signed_direction: None,
                        source_frame_id: Some("frame-cue-only".into()),
                    }),
                    decision_frame_id: None,
                    decision_calibration_revision: None,
                    decision_rectification_revision: None,
                    table_visibility: None,
                    table_projection: None,
                },
            )
            .unwrap();

        let published = take_latest_state(&mut latest).unwrap();
        assert_eq!(published.current.payload.balls.len(), 0);
        assert_eq!(published.current.localization_received_monotonic_ns, None);
        assert_eq!(
            published.current.payload.observation_status["ball_positions_status"],
            "unavailable"
        );
        let websocket = websocket_scene_message(&published);
        assert_eq!(websocket["data"]["ball_count"], 0);
        assert_eq!(
            websocket["data"]["observation_status"]["perception"]["phase"],
            "pre_shot"
        );
        assert_eq!(
            websocket["data"]["observation_status"]["perception"]["cue"]["shaft_start_px"][0],
            100.0
        );

        let (ready_status, ready) = get_json(gateway.router(), "/api/v1/health/ready").await;
        assert_eq!(ready_status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(ready["localization_available"], false);
    }

    #[tokio::test]
    async fn live_v3_rejects_stale_acknowledgement_before_ready() {
        let (directory, configuration_path) = live_test_configuration();
        let gateway = Gateway::new(ServerConfig {
            configuration_path: Some(configuration_path),
            project_root: Some(directory.path().to_path_buf()),
            ..test_config()
        })
        .unwrap();
        let settings = gateway.resolve_live_capture_settings().unwrap();
        let address = start_ingress(gateway.clone()).await;
        let mut stream = TcpStream::connect(address).await.unwrap();
        let mut acknowledgement = live_acknowledgement(&settings, "capture-session-stale");
        acknowledgement["configuration_revision"] =
            json!(settings.configuration_revision.saturating_add(1));
        send_frame(
            &mut stream,
            &serde_json::to_vec(&json!({
                "type": "hello",
                "protocol_version": LIVE_PROTOCOL_VERSION,
                "engine_instance_id": "engine-live-stale",
                "source_session_id": "capture-session-stale",
                "capture_session_ack": acknowledgement,
            }))
            .unwrap(),
        )
        .await;

        let mut response_byte = [0_u8; 1];
        let read = tokio::time::timeout(Duration::from_secs(1), stream.read(&mut response_byte))
            .await
            .expect("rejected live handshake should close promptly")
            .unwrap();
        assert_eq!(read, 0, "invalid readiness must not receive a ready frame");
        assert!(gateway.snapshot_now().is_none());
        gateway.shutdown();
    }

    #[test]
    fn live_gateway_requires_configured_capture_settings() {
        let gateway = Gateway::new(test_config()).unwrap();
        assert!(gateway
            .resolve_live_capture_settings()
            .unwrap_err()
            .contains("--config"));
    }

    #[tokio::test]
    async fn shutdown_closes_an_active_blocking_producer_reader() {
        let gateway = Gateway::new(test_config()).unwrap();
        let address = start_ingress(gateway.clone()).await;
        let mut producer = TcpStream::connect(address).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !gateway.inner.producer_connected.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        gateway.shutdown();

        let mut byte = [0_u8; 1];
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), producer.read(&mut byte))
                .await
                .unwrap()
                .unwrap_or(0),
            0
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while gateway.inner.producer_connected.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn identity_restart_clears_old_localization_and_starts_new_sequence_epoch() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 4, "frame-a"))
            .unwrap();
        assert!(gateway.snapshot_now().is_some());

        gateway
            .activate_producer_epoch("engine-b".into(), "session-b".into())
            .unwrap();
        assert!(gateway.snapshot_now().is_none());
        gateway
            .accept_scene(scene_for("engine-b", "session-b", 1, "frame-1"))
            .unwrap();
        let state = gateway.snapshot_now().unwrap();
        assert_eq!(
            state.current.payload.observation_status["engine_instance_id"],
            "engine-b"
        );
    }

    #[tokio::test]
    async fn lifecycle_rejects_invalid_identity_order_and_duplicate_observation() {
        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        let wrong_identity = gateway
            .accept_scene(scene_for("wrong-engine", "session-a", 1, "frame-1"))
            .unwrap_err();
        assert!(wrong_identity.contains("active source"));
        assert!(gateway.snapshot_now().is_none());

        gateway
            .activate_producer_epoch("engine-b".into(), "session-b".into())
            .unwrap();
        let first = scene_for("engine-b", "session-b", 1, "frame-1");
        gateway.accept_scene(first).unwrap();
        let duplicate_sequence = scene_for("engine-b", "session-b", 1, "frame-2");
        let duplicate = gateway.accept_scene(duplicate_sequence).unwrap_err();
        assert!(duplicate.contains("sequence"));
        let stale_sequence = scene_for("engine-b", "session-b", 2, "frame-1");
        let stale_frame = gateway.accept_scene(stale_sequence).unwrap_err();
        assert!(stale_frame.contains("already accepted"));
        gateway
            .accept_scene(scene_for("engine-b", "session-b", 2, "frame-2"))
            .unwrap();
        assert_eq!(
            gateway
                .snapshot_now()
                .unwrap()
                .current
                .payload
                .observation_status["source_frame_id"],
            "frame-2"
        );
    }

    #[tokio::test]
    async fn tcp_read_timeout_and_oversize_frames_are_rejected_without_state() {
        let config = ServerConfig {
            maximum_scene_bytes: 4096,
            tcp_read_timeout: Duration::from_millis(80),
            ..test_config()
        };
        let gateway = Gateway::new(config.clone()).unwrap();
        let address = start_ingress(gateway.clone()).await;
        let mut stalled = TcpStream::connect(address).await.unwrap();
        let mut buffer = [0; 1];
        let timeout_result =
            tokio::time::timeout(Duration::from_millis(500), stalled.read(&mut buffer)).await;
        assert!(timeout_result.unwrap().unwrap_or(0) == 0);

        let mut oversized = TcpStream::connect(address).await.unwrap();
        oversized
            .write_all(&((config.maximum_scene_bytes as u32 + 1).to_be_bytes()))
            .await
            .unwrap();
        let read = tokio::time::timeout(Duration::from_secs(1), oversized.read(&mut buffer)).await;
        assert_eq!(read.unwrap().unwrap_or(0), 0);
        assert!(gateway.snapshot_now().is_none());
    }

    #[tokio::test]
    async fn latest_watch_skips_history_and_send_deadline_expires_a_stalled_write() {
        let (sender, mut receiver) = watch::channel(Some(0_u64));
        sender.send_replace(Some(1));
        sender.send_replace(Some(2));
        receiver.changed().await.unwrap();
        assert_eq!(*receiver.borrow_and_update(), Some(2));

        let send = within_send_deadline(
            Duration::from_millis(10),
            std::future::pending::<Result<(), ()>>(),
        )
        .await;
        assert!(send.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_real_slow_websocket_is_closed_without_blocking_http_or_other_clients() {
        let websocket_send_deadline = Duration::from_secs(2);
        let config = ServerConfig {
            maximum_scene_bytes: 32 * 1024 * 1024,
            tcp_read_timeout: Duration::from_secs(10),
            stale_after: Duration::from_secs(30),
            expire_after: Duration::from_secs(60),
            websocket_send_deadline,
            ..test_config()
        };
        let gateway = Gateway::new(config).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 1, "frame-1"))
            .unwrap();

        let http_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_address = http_listener.local_addr().unwrap();
        let app = gateway.router();
        tokio::spawn(async move { axum::serve(http_listener, app).await.unwrap() });

        // Constrain only this client's receive window. It consumes the initial small
        // state; a single valid, oversized public state then forces the real server
        // WebSocket writer to wait for socket capacity and hit its configured deadline.
        let slow_socket = Socket::new(Domain::IPV4, Type::STREAM, Some(Protocol::TCP)).unwrap();
        slow_socket.set_recv_buffer_size(1_024).unwrap();
        slow_socket.connect(&http_address.into()).unwrap();
        slow_socket.set_nonblocking(true).unwrap();
        let slow_tcp: std::net::TcpStream = slow_socket.into();
        let slow_tcp = TcpStream::from_std(slow_tcp).unwrap();
        let slow_request = format!("ws://{http_address}/api/v1/websocket/ws")
            .into_client_request()
            .unwrap();
        let test_websocket_config = WebSocketConfig::default()
            .max_message_size(Some(64 * 1024 * 1024))
            .max_frame_size(Some(64 * 1024 * 1024));
        let (mut slow_client, _) =
            client_async_with_config(slow_request, slow_tcp, Some(test_websocket_config))
                .await
                .unwrap();
        let initial = tokio::time::timeout(Duration::from_secs(1), slow_client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(matches!(initial, ClientMessage::Text(_)));

        let (mut responsive_client, _) = connect_async_with_config(
            format!("ws://{http_address}/api/v1/websocket/ws"),
            Some(test_websocket_config),
            false,
        )
        .await
        .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(1), responsive_client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        let oversized_sequence = 2_u64;
        let responsive_reader = tokio::spawn(async move {
            loop {
                let message =
                    tokio::time::timeout(Duration::from_secs(10), responsive_client.next())
                        .await
                        .expect("responsive websocket should keep receiving states")
                        .expect("responsive websocket should remain open")
                        .expect("responsive websocket message should decode");
                if let ClientMessage::Text(encoded) = message {
                    let state: Value = serde_json::from_str(&encoded).unwrap();
                    if state["data"]["frame_number"].as_u64() == Some(oversized_sequence) {
                        break (responsive_client, state);
                    }
                }
            }
        });
        gateway
            .accept_scene(oversized_public_state_scene(oversized_sequence))
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let published = gateway.snapshot_now().is_some_and(|snapshot| {
                    transport_sequence(&snapshot.current.payload.observation_status)
                        == Some(oversized_sequence)
                        && snapshot.current.payload.balls.len() == 64
                        && snapshot.current.payload.balls[0].identity_evidence.len() == 2_048
                });
                if published {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("oversized scene should publish to the state lifecycle");
        let (mut responsive_client, latest_fast_state) = responsive_reader.await.unwrap();
        assert_eq!(
            latest_fast_state["data"]["frame_number"],
            oversized_sequence
        );
        assert_eq!(latest_fast_state["data"]["ball_count"], 64);

        // One complete state may still fit the kernel's autotuned send queue on some
        // systems. A second same-sized latest update forces the unread socket to its
        // real transport limit while the responsive client keeps consuming.
        let queued_sequence = oversized_sequence + 1;
        gateway
            .accept_scene(oversized_public_state_scene(queued_sequence))
            .unwrap();
        let queued_fast_message =
            tokio::time::timeout(Duration::from_secs(10), responsive_client.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        let ClientMessage::Text(queued_fast_message) = queued_fast_message else {
            panic!("responsive websocket should receive the next latest state")
        };
        assert_eq!(
            serde_json::from_str::<Value>(&queued_fast_message).unwrap()["data"]["frame_number"],
            queued_sequence
        );
        let published_at = Instant::now();
        // Leave the slow socket completely unread while the server has the full
        // configured deadline to encounter backpressure. Reading even one frame would
        // restore its TCP window and could let the send complete.
        tokio::time::sleep(websocket_send_deadline + Duration::from_millis(100)).await;

        let slow_closed = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match slow_client.next().await {
                    None | Some(Err(_)) | Some(Ok(ClientMessage::Close(_))) => return true,
                    Some(Ok(_)) => {}
                }
            }
        })
        .await
        .unwrap_or(false);
        assert!(
            slow_closed,
            "slow websocket stayed connected past its deadline"
        );
        assert!(
            published_at.elapsed()
                >= websocket_send_deadline.saturating_sub(Duration::from_millis(15)),
            "slow peer closed before the configured send deadline elapsed"
        );

        let (health_status, health) = get_json(gateway.router(), "/api/v1/health/live").await;
        assert_eq!(health_status, StatusCode::OK);
        assert_eq!(health["status"], "alive");

        let next_sequence = queued_sequence + 1;
        gateway.accept_scene(dense_scene(next_sequence)).unwrap();
        let next = tokio::time::timeout(Duration::from_secs(1), responsive_client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let ClientMessage::Text(next) = next else {
            panic!("responsive websocket should remain connected")
        };
        assert_eq!(
            serde_json::from_str::<Value>(&next).unwrap()["data"]["frame_number"],
            next_sequence
        );
        let (status, state) = get_json(gateway.router(), "/api/v1/game/state").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(state["frame_number"], next_sequence);
    }

    #[tokio::test]
    async fn websocket_upgrade_is_explicitly_unavailable_before_first_localization() {
        let gateway = Gateway::new(test_config()).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = gateway.router();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let error = tokio_tungstenite::connect_async(format!("ws://{address}/api/v1/websocket/ws"))
            .await
            .unwrap_err();
        let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
            panic!("expected HTTP not-ready response, got {error}")
        };
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn diagnostics_websocket_starts_without_scene_and_pushes_increasing_revisions() {
        let mut config = test_config();
        config.maximum_websocket_clients = 1;
        let gateway =
            Gateway::new_with_audit_and_diagnostics(config, None, diagnostics_test_policy())
                .unwrap();
        let snapshot_gateway = gateway.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, gateway.router()).await.unwrap();
        });
        let url = format!("ws://{address}{DIAGNOSTICS_WEBSOCKET_PATH}");
        let (mut client, _) = connect_async_with_config(&url, None, false).await.unwrap();
        let initial = receive_diagnostics_snapshot(&mut client).await;
        assert_eq!(initial["schema_version"], 1);
        assert_eq!(initial["producer"]["state"], "unknown");
        assert_eq!(initial["producer"]["metadata"], Value::Null);
        assert_eq!(initial["stages"].as_array().unwrap().len(), 7);
        assert_eq!(initial["stages"][5]["status"], "starting");
        assert_eq!(initial["stages"][6]["status"], "unknown");

        let second = connect_async_with_config(&url, None, false)
            .await
            .unwrap_err();
        let tokio_tungstenite::tungstenite::Error::Http(response) = second else {
            panic!("expected bounded diagnostics-client rejection, got {second}")
        };
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        // The router owns a watch receiver, so a full replacement snapshot is
        // sent without reopening the connection or publishing scene data.
        // This direct push also keeps the route test independent of timers.
        // The publisher uses this same method for evidence/heartbeat updates.
        snapshot_gateway.publish_diagnostics_snapshot();
        let updated = receive_diagnostics_snapshot(&mut client).await;
        assert!(
            updated["gateway"]["revision"].as_u64().unwrap()
                > initial["gateway"]["revision"].as_u64().unwrap()
        );
        drop(client);
        server.abort();
    }

    #[tokio::test]
    async fn diagnostics_publisher_expires_reports_without_new_packets() {
        use crate::producer_diagnostics_contract::{
            ProducerDiagnosticsReport, ProducerStageName, StageObservation,
        };

        let policy = diagnostics_test_policy();
        let gateway = Gateway::new_with_audit_and_diagnostics(test_config(), None, policy).unwrap();
        let registry = gateway.diagnostics_registry();
        let mut report = ProducerDiagnosticsReport::decode(
            include_bytes!("../contracts/producer-diagnostics/v1/active.json"),
            65_536,
        )
        .unwrap();
        report.generation_id = "expiry-test-generation".into();
        report.sequence = 1;
        report.stages = vec![StageObservation {
            stage: ProducerStageName::CaptureProgress,
            status: StageStatus::Healthy,
            reason_code: "capture_frame_received".into(),
            detail: None,
            last_progress_age_ms: Some(120),
            in_flight: false,
            work_started_age_ms: None,
            expected_jobs: None,
            completed_jobs: None,
            failed_jobs: None,
            scheduler_state: None,
            active_job_kind: None,
            last_completion_age_ms: None,
            last_failure_code: None,
            last_failure_age_ms: None,
            deadline_ms: None,
        }];
        registry.begin_generation(report.generation_id.clone());
        registry.accept(report, Instant::now()).unwrap();

        let mut updates = gateway.inner.diagnostics_snapshots.subscribe();
        let publisher = tokio::spawn(run_diagnostics_snapshot_publisher(gateway.clone()));
        let mut last_revision = updates.borrow_and_update().gateway.revision;
        let fresh = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                updates.changed().await.unwrap();
                let snapshot = updates.borrow_and_update().clone();
                assert!(snapshot.gateway.revision > last_revision);
                last_revision = snapshot.gateway.revision;
                if snapshot.producer.state == ProducerReportState::Fresh {
                    break snapshot;
                }
            }
        })
        .await
        .expect("fresh diagnostics report was not published");
        assert!(fresh.producer.metadata.is_some());

        let expired = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                updates.changed().await.unwrap();
                let snapshot = updates.borrow_and_update().clone();
                assert!(snapshot.gateway.revision > last_revision);
                last_revision = snapshot.gateway.revision;
                if snapshot.producer.state == ProducerReportState::Expired {
                    break snapshot;
                }
            }
        })
        .await
        .expect("report expiry did not publish without another producer packet");
        assert!(expired.producer.metadata.is_none());
        assert!(expired.producer.report_age_ms.is_some());
        assert_eq!(expired.producer.expires_in_ms, None);
        let last_known = expired.producer.last_known.as_ref().unwrap();
        assert_eq!(last_known.report.generation_id, "expiry-test-generation");
        assert!(last_known.report.metadata.model.is_some());
        assert!(!last_known.report.stages.is_empty());
        for stage in expired.stages.iter().filter(|stage| {
            matches!(
                stage.stage,
                DiagnosticsStageName::SourceLifecycle
                    | DiagnosticsStageName::CaptureProgress
                    | DiagnosticsStageName::ModelInitialization
                    | DiagnosticsStageName::ScheduledPerceptionWork
            )
        }) {
            assert_eq!(stage.status, StageStatus::Unknown);
        }
        gateway.shutdown();
        publisher.await.unwrap();
    }

    #[test]
    fn active_job_deadline_stalls_only_inflight_work_and_phase_gated_work_stays_idle() {
        use crate::producer_diagnostics_contract::{
            ProducerDiagnosticsReport, ProducerStageName, ScheduledJobKind, ScheduledWorkState,
            StageObservation,
        };

        let gateway =
            Gateway::new_with_audit_and_diagnostics(test_config(), None, diagnostics_test_policy())
                .unwrap();
        let registry = gateway.diagnostics_registry();
        let mut report = ProducerDiagnosticsReport::decode(
            include_bytes!("../contracts/producer-diagnostics/v1/active.json"),
            65_536,
        )
        .unwrap();
        report.generation_id = "active-work-generation".into();
        report.sequence = 1;
        report.stages = vec![StageObservation {
            stage: ProducerStageName::ScheduledPerceptionWork,
            status: StageStatus::Healthy,
            reason_code: "scheduled_job_in_flight".into(),
            detail: None,
            last_progress_age_ms: Some(5),
            in_flight: true,
            work_started_age_ms: Some(60),
            expected_jobs: Some(12),
            completed_jobs: Some(11),
            failed_jobs: Some(0),
            scheduler_state: Some(ScheduledWorkState::InFlight),
            active_job_kind: Some(ScheduledJobKind::BallDetection),
            last_completion_age_ms: Some(50),
            last_failure_code: None,
            last_failure_age_ms: None,
            deadline_ms: Some(50),
        }];
        registry.begin_generation(report.generation_id.clone());
        registry.accept(report.clone(), Instant::now()).unwrap();
        let overdue = gateway.assess_diagnostics_snapshot(2);
        let work = overdue
            .stages
            .iter()
            .find(|stage| stage.stage == DiagnosticsStageName::ScheduledPerceptionWork)
            .unwrap();
        assert_eq!(work.status, StageStatus::Stalled);
        assert_eq!(work.reason_code, "active_work_deadline_exceeded");
        assert_eq!(work.active_job_kind, Some(ScheduledJobKind::BallDetection));

        report.sequence += 1;
        report.stages[0].status = StageStatus::Idle;
        report.stages[0].reason_code = "work_phase_gated".into();
        report.stages[0].in_flight = false;
        report.stages[0].work_started_age_ms = None;
        report.stages[0].active_job_kind = None;
        report.stages[0].scheduler_state = Some(ScheduledWorkState::PhaseGated);
        report.stages[0].deadline_ms = None;
        registry.accept(report, Instant::now()).unwrap();
        let idle = gateway.assess_diagnostics_snapshot(3);
        let work = idle
            .stages
            .iter()
            .find(|stage| stage.stage == DiagnosticsStageName::ScheduledPerceptionWork)
            .unwrap();
        assert_eq!(work.status, StageStatus::Idle);
        assert_eq!(work.scheduler_state, Some(ScheduledWorkState::PhaseGated));
        assert_eq!(work.deadline_ms, None);
    }

    #[test]
    fn supervised_exit_outcomes_distinguish_graceful_stop_from_failures() {
        use crate::producer_diagnostics_contract::ProducerExitReason;

        let gateway =
            Gateway::new_with_audit_and_diagnostics(test_config(), None, diagnostics_test_policy())
                .unwrap();
        let registry = gateway.diagnostics_registry();
        for (index, outcome, expected_status, expected_reason) in [
            (
                1,
                ProducerExitReason::GracefulStop,
                StageStatus::Stopped,
                "producer_graceful_stop",
            ),
            (
                2,
                ProducerExitReason::ProcessExitedNonzero,
                StageStatus::Failed,
                "producer_process_exited_nonzero",
            ),
            (
                3,
                ProducerExitReason::SpawnFailed,
                StageStatus::Failed,
                "producer_spawn_failed",
            ),
        ] {
            let generation = format!("exit-outcome-{index}");
            registry.begin_generation(generation.clone());
            registry.retire_generation(&generation, outcome);
            let snapshot = gateway.assess_diagnostics_snapshot(index + 1);
            let source = snapshot
                .stages
                .iter()
                .find(|stage| stage.stage == DiagnosticsStageName::SourceLifecycle)
                .unwrap();
            assert_eq!(source.status, expected_status);
            assert_eq!(source.reason_code, expected_reason);
            assert_eq!(snapshot.producer.exit_reason, Some(outcome));
            assert!(snapshot.producer.metadata.is_none());
        }
    }
}

#[cfg(test)]
mod native_outline_delivery_tests {
    use super::*;
    #[test]
    fn public_outline_requires_exact_observed_source_geometry() {
        let mut snapshot = crate::decode_scene_snapshot_v1(include_bytes!(
            "../contracts/v1/scene-rust-tracked-success.json"
        ))
        .unwrap();
        let measurement = snapshot.balls[0].measurement.as_mut().unwrap();
        measurement.image_center_px = crate::Point2d { x: 2., y: 2. };
        measurement.image_radius_px = Some(crate::Point2d { x: 6., y: 3. });
        measurement.image_outline_px = Some(vec![
            crate::Point2d { x: -4., y: 2. },
            crate::Point2d { x: 2., y: -1. },
            crate::Point2d { x: 8., y: 2. },
            crate::Point2d { x: 2., y: 5. },
        ]);
        let payload =
            crate::public_state::SceneSnapshotStateAdapter::new(PublicStateConfig::default())
                .unwrap()
                .adapt(&snapshot, None, None)
                .unwrap()
                .unwrap();
        let ball = &payload.balls[0];
        validate_live_public_ball(ball, &snapshot).unwrap();
        let mut forged = ball.clone();
        forged.outline.as_mut().unwrap().points[0].x += 1.;
        assert!(validate_live_public_ball(&forged, &snapshot).is_err());
        let mut stale = ball.clone();
        stale.is_observed = false;
        assert!(validate_live_public_ball(&stale, &snapshot).is_err());
        let mut wrong_frame = ball.clone();
        wrong_frame.source_observation.frame_id = "different-frame".into();
        assert!(validate_live_public_ball(&wrong_frame, &snapshot).is_err());
        let mut wrong_scale = ball.clone();
        wrong_scale.outline.as_mut().unwrap().scale = [2., 2.];
        assert!(validate_live_public_ball(&wrong_scale, &snapshot).is_err());
    }
}
