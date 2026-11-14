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

use axum::body::Body;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, Query, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, Mutex as AsyncMutex, Semaphore};
use tokio::time::{sleep, timeout};

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
const IDENTIFIED_MJPEG_BOUNDARY: &str = "billiards-frame-v1";
const IDENTIFIED_MJPEG_PATH: &str = "/api/v1/video/identified.mjpg";
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
    producer_control_stream: Mutex<Option<std::net::TcpStream>>,
    websocket_slots: Arc<Semaphore>,
    configuration_slots: Arc<Semaphore>,
    frame_preview_stream: Option<FramePreviewStreamSettings>,
    frame_preview_stream_slots: Arc<Semaphore>,
    configuration_mutation_lock: Arc<AsyncMutex<()>>,
    gateway_audit: Option<Arc<GatewayAudit>>,
}

#[derive(Debug, Clone)]
struct FramePreviewStreamSettings {
    address: SocketAddr,
    request_timeout: Duration,
    poll_interval: Duration,
    maximum_header_bytes: usize,
    maximum_jpeg_bytes: usize,
    maximum_clients: usize,
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
        config.validate()?;
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
        let frame_preview_stream_slots = Arc::new(Semaphore::new(
            frame_preview_stream
                .as_ref()
                .map_or(1, |settings| settings.maximum_clients),
        ));
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
        let gateway_audit = audit.map(|(recorder, run_id)| {
            Arc::new(GatewayAudit {
                recorder,
                run_id,
                gateway_instance_id: uuid::Uuid::new_v4().to_string(),
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
                frame_preview_stream_slots,
                configuration_mutation_lock: Arc::new(AsyncMutex::new(())),
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
                }),
                latest,
                shutdown,
                producer_connected: AtomicBool::new(false),
                producer_control_stream: Mutex::new(None),
                gateway_audit,
            }),
        })
    }

    pub fn router(&self) -> Router {
        let scene_router = Router::new()
            .route("/api/v1/health/live", get(health_live))
            .route("/api/v1/health/ready", get(health_ready))
            .route(IDENTIFIED_MJPEG_PATH, get(identified_mjpeg_stream))
            .route("/api/v1/websocket/ws", get(websocket_upgrade))
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
        let mut state = lock_unpoisoned(&self.inner.state);
        if let Some((width_px, height_px)) = state.live_source_geometry {
            if (snapshot.source.width_px, snapshot.source.height_px) != (width_px, height_px) {
                return Err("scene dimensions differ from acknowledged capture geometry".into());
            }
        }
        let now_ns = self.monotonic_ns();
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
        if outcome == IngestOutcome::AnnotationIgnored {
            return Ok(());
        }

        if outcome == IngestOutcome::LocalizationUpdated {
            state.localization_received_at = Some(received_at);
        }
        state.updated_at = Some(received_at);
        let current = state
            .lifecycle
            .current(now_ns)
            .map_err(|error| error.to_string())?;
        if let Some(current) = current {
            if current.localization_received_monotonic_ns.is_some() {
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
        lock_unpoisoned(&self.inner.state).live_source_geometry =
            Some((geometry.width_px(), geometry.height_px()));
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
            self.accept_scene_with_capture_age(
                snapshot,
                localization_age_offset_ns,
                ball_age_offsets_ns,
                table,
                game_type,
            )?;
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
        let status = serde_json::to_value(perception)
            .map_err(|error| format!("could not encode perception status: {error}"))?;
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
        self.0.producer_connected.store(false, Ordering::Release);
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
    let maximum_clients = positive_size("frame_preview_max_concurrent_encodes")?;
    let framerate = configuration
        .pointer("/api/stream/framerate")
        .and_then(Value::as_object)
        .ok_or_else(|| "api.stream.framerate is missing or invalid".to_owned())?;
    let minimum_fps = framerate
        .get("min_fps")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| "api.stream.framerate.min_fps must be positive".to_owned())?;
    let maximum_fps = framerate
        .get("max_fps")
        .and_then(Value::as_u64)
        .filter(|value| *value >= minimum_fps)
        .ok_or_else(|| "api.stream.framerate.max_fps is invalid".to_owned())?;
    let fps = framerate
        .get("default_fps")
        .and_then(Value::as_u64)
        .filter(|value| (minimum_fps..=maximum_fps).contains(value))
        .ok_or_else(|| "api.stream.framerate.default_fps is invalid".to_owned())?;
    Ok(Some(FramePreviewStreamSettings {
        address: SocketAddr::new(bind_address, port),
        request_timeout: Duration::from_millis(request_timeout_ms),
        poll_interval: Duration::from_secs_f64(1.0 / fps as f64),
        maximum_header_bytes: positive_size("frame_preview_max_header_bytes")?,
        maximum_jpeg_bytes: positive_size("frame_preview_max_jpeg_bytes")?,
        maximum_clients,
    }))
}

async fn identified_mjpeg_stream(State(gateway): State<Gateway>) -> Response {
    let Some(settings) = gateway.inner.frame_preview_stream.clone() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "identity-preserving frame preview is not configured",
        )
            .into_response();
    };
    let permit = match Arc::clone(&gateway.inner.frame_preview_stream_slots).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "identity-preserving frame preview is at capacity",
            )
                .into_response();
        }
    };
    let initial_state = (
        gateway,
        settings,
        permit,
        None::<String>,
        tokio::time::Instant::now(),
    );
    let stream = futures_util::stream::unfold(initial_state, |state| async move {
        let (gateway, settings, permit, mut last_frame_id, mut next_poll) = state;
        loop {
            if *gateway.inner.shutdown.borrow() {
                return None;
            }
            if next_poll > tokio::time::Instant::now() {
                tokio::time::sleep_until(next_poll).await;
            }
            if *gateway.inner.shutdown.borrow() {
                return None;
            }
            next_poll = tokio::time::Instant::now() + settings.poll_interval;
            let mut bytes = None;
            if let Some(frame) =
                request_preview_frame(settings.address, *PREVIEW_REQUEST_V1, &settings).await
            {
                if last_frame_id.as_deref() != Some(frame.descriptor.source_frame_id.as_str()) {
                    last_frame_id = Some(frame.descriptor.source_frame_id.clone());
                    bytes = encode_identified_mjpeg_part(&frame);
                }
            }
            if let Some(bytes) = bytes {
                return Some((
                    Ok::<Vec<u8>, std::convert::Infallible>(bytes),
                    (gateway, settings, permit, last_frame_id, next_poll),
                ));
            }
        }
    });
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        CONTENT_TYPE,
        format!("multipart/x-mixed-replace; boundary={IDENTIFIED_MJPEG_BOUNDARY}")
            .parse()
            .expect("static multipart content type is valid"),
    );
    response.headers_mut().insert(
        CACHE_CONTROL,
        "no-store, no-cache, must-revalidate"
            .parse()
            .expect("static cache control is valid"),
    );
    response
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

fn encode_identified_mjpeg_part(frame: &IdentifiedMjpegFrame) -> Option<Vec<u8>> {
    let descriptor = &frame.descriptor;
    for value in [
        descriptor.source_frame_id.as_str(),
        descriptor.engine_instance_id.as_str(),
        descriptor.capture_session_id.as_str(),
    ] {
        if value.is_empty()
            || value.len() > MAX_IDENTIFIER_CHARACTERS
            || value.chars().any(char::is_control)
        {
            return None;
        }
    }
    let mut part = format!(
        "--{IDENTIFIED_MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nX-Frame-Id: {}\r\nX-Engine-Instance-Id: {}\r\nX-Source-Session-Id: {}\r\nX-Configuration-Revision: {}\r\nX-Calibration-Revision: {}\r\nX-Rectification-Revision: {}\r\n\r\n",
        frame.jpeg.len(),
        descriptor.source_frame_id,
        descriptor.engine_instance_id,
        descriptor.capture_session_id,
        descriptor.configuration_revision,
        descriptor.calibration_revision,
        descriptor.rectification_revision,
    )
    .into_bytes();
    part.extend_from_slice(&frame.jpeg);
    part.extend_from_slice(b"\r\n");
    Some(part)
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

/// Start HTTP and live-v3 loopback scene listeners until shutdown.
pub async fn run(config: ServerConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    run_with_live(config, LiveProcessLaunchOptions::default()).await
}

/// Start the HTTP/WebSocket gateway and, when requested, supervise the Rust
/// capture/inference child that publishes to its loopback scene ingress.
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
    let gateway =
        Gateway::new_with_audit(config.clone(), gateway_audit).map_err(std::io::Error::other)?;
    let scene_listener =
        TcpListener::bind(SocketAddr::new(config.scene_bind, config.scene_port)).await?;
    let http_listener =
        TcpListener::bind(SocketAddr::new(config.http_bind, config.http_port)).await?;
    let scene_address = scene_listener.local_addr()?;
    let http_address = http_listener.local_addr()?;
    eprintln!("billiards Rust gateway HTTP listening on {http_address}");
    eprintln!("live v3 scene ingress listening on {scene_address}");
    let live_process = if live_options.enabled {
        let authority = gateway
            .inner
            .configuration_authority
            .as_ref()
            .ok_or_else(|| std::io::Error::other("live detector startup requires --config"))?;
        let effective_configuration = authority
            .effective_configuration()
            .map_err(std::io::Error::other)?;
        let project_root = config
            .project_root
            .clone()
            .unwrap_or(std::env::current_dir()?);
        Some(
            LiveProcessConfig::from_effective_configuration(
                &effective_configuration,
                &project_root,
                scene_address,
                live_options.executable_path.as_deref(),
                live_options.capture_helper_path.as_deref(),
                live_options.audit_config,
            )
            .map_err(std::io::Error::other)?,
        )
    } else {
        None
    };

    let ingress_gateway = gateway.clone();
    let ingress = tokio::spawn(async move {
        ingress_gateway
            .accept_scene_connections(scene_listener)
            .await
    });
    let freshness = tokio::spawn(run_freshness_scheduler(gateway.clone()));
    let live_supervisor = live_process.map(|process| {
        let shutdown = gateway.inner.shutdown.subscribe();
        tokio::spawn(process.supervise(shutdown))
    });
    let shutdown_gateway = gateway.clone();
    let server = axum::serve(http_listener, gateway.router()).with_graceful_shutdown(async move {
        shutdown_signal().await;
        shutdown_gateway.shutdown();
    });
    let server_result = server.await;
    gateway.shutdown();
    freshness.abort();
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
            let gateway = self.clone();
            let standard_stream = match stream.into_std() {
                Ok(stream) => stream,
                Err(_) => {
                    self.inner
                        .producer_connected
                        .store(false, Ordering::Release);
                    continue;
                }
            };
            let control_stream = match standard_stream.try_clone() {
                Ok(stream) => stream,
                Err(_) => {
                    self.inner
                        .producer_connected
                        .store(false, Ordering::Release);
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
    use axum::http::Request;
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
    async fn identified_mjpeg_route_forwards_capture_identity_with_the_jpeg() {
        let preview_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let preview_address = preview_listener.local_addr().unwrap();
        let (directory, configuration_path) = live_test_configuration();
        let mut document: Value =
            serde_json::from_slice(&std::fs::read(&configuration_path).unwrap()).unwrap();
        document["perception"]["rust_live"]["frame_preview_enabled"] = json!(true);
        document["perception"]["rust_live"]["frame_preview_bind_address"] = json!("127.0.0.1");
        document["perception"]["rust_live"]["frame_preview_port"] = json!(preview_address.port());
        std::fs::write(&configuration_path, serde_json::to_vec(&document).unwrap()).unwrap();
        let gateway = Gateway::new(ServerConfig {
            http_port: 0,
            scene_port: 0,
            configuration_path: Some(configuration_path),
            ..test_config()
        })
        .unwrap();

        let preview_server = tokio::spawn(async move {
            let (mut socket, _) = preview_listener.accept().await.unwrap();
            let mut request = [0_u8; 8];
            socket.read_exact(&mut request).await.unwrap();
            assert_eq!(&request, PREVIEW_REQUEST_V1);
            let descriptor = FramePreviewDescriptorV1 {
                schema_version: 1,
                engine_instance_id: "engine-a".to_owned(),
                capture_session_id: "session-a".to_owned(),
                source_frame_id: "frame-42".to_owned(),
                width_px: 640,
                height_px: 480,
                pixel_format: "rgb8".to_owned(),
                image_space: crate::detection_batch::ImageSpace::Raw,
                camera_profile_id: "camera-profile".to_owned(),
                configuration_revision: 4,
                calibration_revision: 2,
                rectification_revision: 3,
                source_timestamp: None,
                source_timestamp_semantics:
                    crate::capture::CaptureTimestampSemantics::ReadCompletionOnly,
                host_read_completed: crate::frame_preview::PreviewHostTimestampV1 {
                    ticks: "123".to_owned(),
                    unit: crate::detection_batch::TimeUnit::Nanoseconds,
                    clock: crate::detection_batch::ClockKind::Monotonic,
                    domain_id: "engine-a".to_owned(),
                    stage: crate::detection_batch::TimestampStage::CameraReadComplete,
                    semantics: "read_completion_only".to_owned(),
                },
            };
            let jpeg = [0xff, 0xd8, 0xff, 0xd9];
            let header = FramePreviewResponseHeaderV1 {
                schema_version: 1,
                status: FramePreviewStatusV1::Ok,
                descriptor: Some(descriptor),
                jpeg_bytes: jpeg.len() as u32,
                error: None,
            };
            let header = serde_json::to_vec(&header).unwrap();
            socket
                .write_all(&(header.len() as u32).to_be_bytes())
                .await
                .unwrap();
            socket.write_all(&header).await.unwrap();
            socket.write_all(&jpeg).await.unwrap();
        });

        let response = gateway
            .router()
            .oneshot(
                Request::builder()
                    .uri(IDENTIFIED_MJPEG_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[CONTENT_TYPE],
            "multipart/x-mixed-replace; boundary=billiards-frame-v1"
        );
        let first_part = response
            .into_body()
            .into_data_stream()
            .next()
            .await
            .unwrap()
            .unwrap();
        let first_part = String::from_utf8_lossy(&first_part);
        assert!(first_part.contains("X-Frame-Id: frame-42\r\n"));
        assert!(first_part.contains("X-Engine-Instance-Id: engine-a\r\n"));
        assert!(first_part.contains("X-Source-Session-Id: session-a\r\n"));
        assert!(first_part.contains("X-Configuration-Revision: 4\r\n"));
        assert!(first_part.contains("X-Calibration-Revision: 2\r\n"));
        assert!(first_part.contains("X-Rectification-Revision: 3\r\n"));
        preview_server.await.unwrap();
        drop(directory);
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

        gateway
            .accept_perception_state(
                3,
                "engine-retain",
                "session-retain",
                test_perception_state(3, true, true),
            )
            .unwrap();
        let verified_empty = receive_websocket_state(&mut client).await;
        assert_eq!(verified_empty["data"]["ball_count"], 0);
        assert_eq!(
            verified_empty["data"]["observation_status"]["perception"]["empty_layout_verified"],
            true
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
