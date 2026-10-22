//! Side-by-side Rust HTTP/WebSocket gateway for replay and live scene snapshots.
//!
//! Replay-v2 works without runtime configuration. Live-v3 is accepted only
//! when the gateway can resolve the active Rust-owned capture settings and
//! validate the producer's request-bound readiness acknowledgement. The
//! optional live mode supervises the Rust detector directly; the browser
//! remains HTML/JavaScript/CSS and keeps its ordinary JSON-over-HTTP/WebSocket
//! contract.

use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::{watch, Mutex as AsyncMutex, Semaphore};
use tokio::time::{sleep, timeout};

use crate::capture_settings_wire::{parse_capture_settings_json, ResolvedCaptureSettings};
use crate::configuration_authority::{AuthorityLimits, ConfigurationAuthority};
use crate::framing::{read_framed_message, write_framed_message};
use crate::game_session::GameSessionStore;
use crate::game_session_http::GameSessionHttpState;
use crate::live_handshake::{validate_live_v3_hello, ValidatedLiveHandshake};
use crate::live_supervisor::LiveProcessConfig;
use crate::public_state::PublicStateConfig;
use crate::public_state_lifecycle::{
    CurrentSceneState, IngestOutcome, SceneStateLifecycle, SourceStateMetadata, StateFreshness,
    StateLifecycleConfig,
};
use crate::{decode_scene_snapshot_v1, SceneSnapshotV1};

const REPLAY_PROTOCOL_VERSION: u8 = 2;
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
    /// Active unified config. Replay can run without it; live-v3 readiness cannot.
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

/// Optional live detector launch policy kept separate from replay-only server
/// settings, so tests and replay tools remain independent of camera hardware.
#[derive(Debug, Clone, Default)]
pub struct LiveProcessLaunchOptions {
    pub enabled: bool,
    pub executable_path: Option<PathBuf>,
    pub capture_helper_path: Option<PathBuf>,
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
        if self.maximum_scene_bytes < maximum_v2_handshake_size()
            || self.maximum_scene_bytes > MAXIMUM_SCENE_BYTES_LIMIT
        {
            return Err(format!(
                "maximum scene bytes must be between {} and {}",
                maximum_v2_handshake_size(),
                MAXIMUM_SCENE_BYTES_LIMIT
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

fn maximum_v2_handshake_size() -> usize {
    let hello = json!({
        "type": "hello",
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "engine_instance_id": "e".repeat(MAX_IDENTIFIER_CHARACTERS),
        "source_session_id": "s".repeat(MAX_IDENTIFIER_CHARACTERS),
    });
    serde_json::to_vec(&hello)
        .expect("fixed handshake is serializable")
        .len()
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
    configuration_mutation_lock: Arc<AsyncMutex<()>>,
}

struct IngressState {
    lifecycle: SceneStateLifecycle,
    producer_identity: Option<(String, String)>,
    last_transport_sequence: Option<u64>,
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
    publication_sequence: u64,
}

impl Gateway {
    pub fn new(config: ServerConfig) -> Result<Self, String> {
        config.validate()?;
        let configuration_authority = config
            .configuration_path
            .as_ref()
            .map(|path| ConfigurationAuthority::new(path, AuthorityLimits::default()))
            .transpose()?;
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
        Ok(Self {
            inner: Arc::new(GatewayInner {
                websocket_slots: Arc::new(Semaphore::new(config.maximum_websocket_clients)),
                configuration_slots: Arc::new(Semaphore::new(
                    config.maximum_configuration_requests,
                )),
                configuration_mutation_lock: Arc::new(AsyncMutex::new(())),
                configuration_authority,
                game_session_store,
                config,
                started_at: Instant::now(),
                state: Mutex::new(IngressState {
                    lifecycle,
                    producer_identity: None,
                    last_transport_sequence: None,
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
            }),
        })
    }

    pub fn router(&self) -> Router {
        let scene_router = Router::new()
            .route("/api/v1/health/live", get(health_live))
            .route("/api/v1/health/ready", get(health_ready))
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
        current.localization_received_monotonic_ns?;
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

    fn acknowledge_producer<W: Write>(
        &self,
        writer: &mut W,
        engine: String,
        session: String,
    ) -> Result<(), String> {
        let watermark = self.producer_watermark(&engine, &session);
        let ready = json!({
            "type": "ready",
            "protocol_version": REPLAY_PROTOCOL_VERSION,
            "last_transport_sequence": watermark,
        });
        write_framed_message(
            writer,
            &serde_json::to_vec(&ready).map_err(|error| error.to_string())?,
            self.inner.config.maximum_scene_bytes,
        )
        .map_err(|error| error.to_string())?;
        // A failed ready write leaves the current epoch and positions intact.
        self.activate_producer_epoch(engine, session)?;
        lock_unpoisoned(&self.inner.state).live_source_geometry = None;
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
        let hello_value: Value = serde_json::from_slice(&hello_bytes)
            .map_err(|error| format!("invalid scene producer hello: {error}"))?;
        match hello_value.get("protocol_version").and_then(Value::as_u64) {
            Some(version) if version == u64::from(REPLAY_PROTOCOL_VERSION) => {
                let hello: ProducerHello = serde_json::from_slice(&hello_bytes)
                    .map_err(|error| format!("invalid replay v2 hello: {error}"))?;
                validate_identifier(&hello.engine_instance_id, "engine instance ID")?;
                validate_identifier(&hello.source_session_id, "source session ID")?;
                if hello.message_type != "hello"
                    || hello.protocol_version != REPLAY_PROTOCOL_VERSION
                {
                    return Err("replay v2 hello type or version is invalid".into());
                }
                let engine_instance_id = hello.engine_instance_id;
                let source_session_id = hello.source_session_id;
                self.acknowledge_producer(
                    &mut stream,
                    engine_instance_id.clone(),
                    source_session_id.clone(),
                )?;

                loop {
                    let bytes =
                        read_framed_message(&mut stream, self.inner.config.maximum_scene_bytes)
                            .map_err(|error| error.to_string())?;
                    self.accept_replay_payload(&bytes, &engine_instance_id, &source_session_id)?;
                }
            }
            Some(version) if version == u64::from(LIVE_PROTOCOL_VERSION) => {
                let capture_settings = self.resolve_live_capture_settings()?;
                let handshake = validate_live_v3_hello(&hello_bytes, &capture_settings)
                    .map_err(|error| error.to_string())?;
                self.acknowledge_live_producer(&mut stream, &handshake)?;

                loop {
                    let bytes =
                        read_framed_message(&mut stream, self.inner.config.maximum_scene_bytes)
                            .map_err(|error| error.to_string())?;
                    self.accept_live_delivery(&bytes, &handshake)?;
                }
            }
            _ => Err("unsupported scene producer protocol version".into()),
        }
    }

    fn accept_live_delivery(
        &self,
        bytes: &[u8],
        handshake: &ValidatedLiveHandshake,
    ) -> Result<(), String> {
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

    /// Accept the two explicitly versioned replay-v2 payload forms.
    ///
    /// Historical producers send raw scene-v1 snapshots. Paced Rust replay
    /// sends the public-state v1 envelope so lifecycle ages and optional
    /// metadata survive delivery. Both are bound to the identity from the
    /// replay hello, and both pass through this gateway's own lifecycle.
    fn accept_replay_payload(
        &self,
        bytes: &[u8],
        engine_instance_id: &str,
        source_session_id: &str,
    ) -> Result<(), String> {
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid replay-v2 scene payload: {error}"))?;
        if value.get("delivery_type").is_none() {
            let snapshot = decode_scene_snapshot_v1(bytes)
                .map_err(|error| format!("invalid scene snapshot: {error}"))?;
            if snapshot.engine_instance_id != engine_instance_id
                || snapshot.source.session_id != source_session_id
            {
                return Err("scene identity differs from the replay-v2 hello".into());
            }
            return self.accept_scene(snapshot);
        }

        let delivery: PublicStateDeliveryInput = serde_json::from_value(value)
            .map_err(|error| format!("invalid replay public-state delivery v1: {error}"))?;
        if delivery.delivery_type != "billiards_public_state_delivery"
            || delivery.delivery_schema_version != 1
        {
            return Err("unsupported replay public-state delivery contract".into());
        }
        let snapshot = delivery.source_snapshot;
        snapshot
            .validate()
            .map_err(|error| format!("invalid replay source scene snapshot: {error}"))?;
        if snapshot.engine_instance_id != engine_instance_id
            || snapshot.source.session_id != source_session_id
        {
            return Err("scene identity differs from the replay-v2 hello".into());
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
        )
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
#[serde(deny_unknown_fields)]
struct ProducerHello {
    #[serde(rename = "type")]
    message_type: String,
    protocol_version: u8,
    engine_instance_id: String,
    source_session_id: String,
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

fn validate_identifier(value: &str, name: &str) -> Result<(), String> {
    if value.is_empty()
        || value.chars().count() > MAX_IDENTIFIER_CHARACTERS
        || value.chars().any(char::is_control)
    {
        return Err(format!("{name} must be a bounded printable identity"));
    }
    Ok(())
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

async fn health_live() -> Json<Value> {
    Json(json!({"status": "alive"}))
}

async fn health_ready(State(gateway): State<Gateway>) -> Response {
    match gateway.snapshot_now() {
        Some(snapshot) if snapshot.current.freshness == StateFreshness::Fresh => Json(json!({
            "status": "ready",
            "localization_available": true,
            "freshness": snapshot.current.freshness,
        }))
        .into_response(),
        Some(snapshot) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "localization_available": true,
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

async fn websocket_upgrade(State(gateway): State<Gateway>, upgrade: WebSocketUpgrade) -> Response {
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
    upgrade
        .max_message_size(max_message_size)
        .max_frame_size(max_message_size)
        .on_upgrade(move |socket| websocket_session(socket, gateway, latest, permit))
        .into_response()
}

async fn websocket_session(
    mut socket: WebSocket,
    gateway: Gateway,
    mut latest: watch::Receiver<Option<Arc<PublishedState>>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
) {
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
        gateway.inner.config.websocket_send_deadline,
    )
    .await
    {
        return;
    }
    loop {
        tokio::select! {
            changed = latest.changed() => {
                if changed.is_err() {
                    return;
                }
                let next = take_latest_state(&mut latest);
                match next {
                    Some(state) => {
                        if !send_websocket_state(&mut socket, &state, gateway.inner.config.websocket_send_deadline).await {
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
    send_deadline: Duration,
) -> bool {
    let message = websocket_scene_message(state);
    let encoded = match serde_json::to_string(&message) {
        Ok(encoded) => encoded,
        Err(_) => return false,
    };
    matches!(
        within_send_deadline(send_deadline, socket.send(Message::Text(encoded.into()))).await,
        Some(Ok(()))
    )
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
    // Preserve the established Python scene_state_to_response predicate. Freshness is
    // independently visible in observation_status; it is not part of this contract field.
    let is_valid = scene_state_is_valid(status);
    json!({
        "timestamp": timestamp_text(state.published_at),
        "frame_number": frame_number,
        "balls": balls,
        "cue": null,
        "table": null,
        "game_type": null,
        "session": null,
        "is_valid": is_valid,
        "confidence": confidence,
        "core_metadata_status": "unavailable",
        "core_metadata_timestamp": null,
        "table_metadata_status": "unavailable",
        "scene_calibration_revision": status.get("calibration_revision").cloned().unwrap_or(Value::Null),
        "table_calibration_revision": null,
        "events": [],
        "confidence_status": if confidence.is_some() { "mean_detector_confidence" } else { "unavailable" },
        "observation_status": status,
        "perception_source": "rust_scene",
    })
}

fn scene_state_is_valid(status: &Value) -> bool {
    // Keep this predicate identical to Python's scene_state_to_response: stale/cached
    // status is exposed in observation_status and also makes the position check fail.
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

/// Start HTTP and replay-v2/live-v3 loopback scene listeners until shutdown.
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
    let gateway = Gateway::new(config.clone()).map_err(std::io::Error::other)?;
    let scene_listener =
        TcpListener::bind(SocketAddr::new(config.scene_bind, config.scene_port)).await?;
    let http_listener =
        TcpListener::bind(SocketAddr::new(config.http_bind, config.http_port)).await?;
    let scene_address = scene_listener.local_addr()?;
    let http_address = http_listener.local_addr()?;
    eprintln!("billiards Rust gateway HTTP listening on {http_address}");
    eprintln!("replay v2 / live v3 scene ingress listening on {scene_address}");
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
    use tower::ServiceExt;

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
        decode_scene_snapshot_v1(include_bytes!(
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

    async fn hello(stream: &mut TcpStream, engine: &str, session: &str) -> Value {
        send_frame(
            stream,
            &serde_json::to_vec(&json!({
                "type":"hello",
                "protocol_version":2,
                "engine_instance_id":engine,
                "source_session_id":session,
            }))
            .unwrap(),
        )
        .await;
        serde_json::from_slice(&read_frame(stream).await).unwrap()
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

    fn replay_public_state_delivery(scene: &SceneSnapshotV1) -> Vec<u8> {
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

    async fn send_scene(stream: &mut TcpStream, scene: &SceneSnapshotV1) {
        send_frame(stream, &serde_json::to_vec(scene).unwrap()).await;
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
                maximum_scene_bytes: 1,
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
    async fn replay_v2_public_state_delivery_reaches_http_with_receiver_local_freshness() {
        let gateway = Gateway::new(test_config()).unwrap();
        let scene_address = start_ingress(gateway.clone()).await;
        let engine = "engine-paced-replay";
        let session = "session-paced-replay";
        let mut producer = TcpStream::connect(scene_address).await.unwrap();
        let ready = hello(&mut producer, engine, session).await;
        assert_eq!(ready["protocol_version"], REPLAY_PROTOCOL_VERSION);

        let scene = scene_for(engine, session, 1, "frame-paced-replay");
        let delivery = replay_public_state_delivery(&scene);
        let sent: Value = serde_json::from_slice(&delivery).unwrap();
        assert_eq!(sent["delivery_type"], "billiards_public_state_delivery");
        assert_eq!(sent["delivery_schema_version"], 1);
        send_frame(&mut producer, &delivery).await;

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
        .expect("public-state replay should become available over HTTP");
        assert_eq!(response["perception_source"], "rust_scene");
        assert_eq!(response["frame_number"], 1);
        assert_eq!(
            response["balls"].as_array().unwrap().len(),
            scene.balls.len()
        );
        assert_eq!(response["observation_status"]["source_session_id"], session);
        assert_eq!(response["observation_status"]["freshness_status"], "fresh");
        producer.shutdown().await.unwrap();
    }

    #[test]
    fn replay_public_state_delivery_requires_v1_and_hello_bound_identity() {
        let gateway = Gateway::new(test_config()).unwrap();
        let scene = scene_for(
            "engine-replay-envelope",
            "session-replay-envelope",
            1,
            "frame-1",
        );
        let mut delivery: Value =
            serde_json::from_slice(&replay_public_state_delivery(&scene)).unwrap();

        delivery["delivery_schema_version"] = json!(2);
        let error = gateway
            .accept_replay_payload(
                &serde_json::to_vec(&delivery).unwrap(),
                &scene.engine_instance_id,
                &scene.source.session_id,
            )
            .unwrap_err();
        assert!(error.contains("unsupported replay public-state delivery contract"));

        delivery["delivery_schema_version"] = json!(1);
        let error = gateway
            .accept_replay_payload(
                &serde_json::to_vec(&delivery).unwrap(),
                "another-engine",
                &scene.source.session_id,
            )
            .unwrap_err();
        assert!(error.contains("scene identity differs from the replay-v2 hello"));

        let error = gateway
            .accept_replay_payload(
                &serde_json::to_vec(&scene).unwrap(),
                "another-engine",
                &scene.source.session_id,
            )
            .unwrap_err();
        assert!(error.contains("scene identity differs from the replay-v2 hello"));
    }

    #[tokio::test]
    async fn replay_v2_scene_is_available_over_http_and_latest_only_websocket() {
        let gateway = Gateway::new(test_config()).unwrap();
        let scene_address = start_ingress(gateway.clone()).await;
        let mut producer = TcpStream::connect(scene_address).await.unwrap();
        let scene = fixture_scene();
        let ready = hello(
            &mut producer,
            &scene.engine_instance_id,
            &scene.source.session_id,
        )
        .await;
        assert_eq!(
            ready,
            json!({"type":"ready","protocol_version":2,"last_transport_sequence":null})
        );
        send_scene(&mut producer, &scene).await;

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
                "radius_unit",
                "radius_source",
                "radius_scale",
                "image_space",
                "image_width_px",
                "image_height_px",
            ],
        );
        let legacy_http_contract: Value = serde_json::from_slice(include_bytes!(
            "../../backend/tests/fixtures/contracts/v1/game_state_http.json"
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
            "../../backend/tests/fixtures/contracts/v1/websocket_state_rust_scene.json"
        ))
        .unwrap();
        let http_scene_projection = json!({
            "ball_count": response["balls"].as_array().unwrap().len(),
            "balls": response["balls"].as_array().unwrap().iter().map(|ball| json!({
                "id": ball["id"],
                "position": ball["position"],
                "table_position": ball["table_position"],
                "radius": ball["radius"],
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
            "../../backend/tests/fixtures/contracts/v1/websocket_state_rust_scene.json"
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

        let mut next_scene = scene_for("engine-a", "session-a", 2, "frame-2");
        next_scene.engine_instance_id = "engine-rust-stage2-a".into();
        next_scene.source.session_id = "replay-session-01".into();
        for ball in &mut next_scene.balls {
            ball.last_observation.session_id = "replay-session-01".into();
        }
        send_scene(&mut producer, &next_scene).await;
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
    fn scene_validity_matches_the_python_converter_predicate() {
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
    fn failed_ready_write_does_not_commit_a_new_producer_epoch() {
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

        let gateway = Gateway::new(test_config()).unwrap();
        gateway
            .activate_producer_epoch("engine-a".into(), "session-a".into())
            .unwrap();
        gateway
            .accept_scene(scene_for("engine-a", "session-a", 1, "frame-1"))
            .unwrap();

        assert!(gateway
            .acknowledge_producer(&mut FailedWriter, "engine-b".into(), "session-b".into(),)
            .is_err());
        let retained = gateway.snapshot_now().unwrap();
        assert_eq!(
            retained.current.payload.observation_status["engine_instance_id"],
            "engine-a"
        );

        let mut ready = Vec::new();
        gateway
            .acknowledge_producer(&mut ready, "engine-b".into(), "session-b".into())
            .unwrap();
        assert!(gateway.snapshot_now().is_none());
        let length = u32::from_be_bytes(ready[..4].try_into().unwrap()) as usize;
        let response: Value = serde_json::from_slice(&ready[4..4 + length]).unwrap();
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
    fn replay_only_gateway_refuses_to_resolve_live_capture_settings() {
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
        hello(&mut producer, "engine-a", "session-a").await;
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
        let address = start_ingress(gateway.clone()).await;
        let mut first = TcpStream::connect(address).await.unwrap();
        hello(&mut first, "engine-a", "session-a").await;
        send_scene(
            &mut first,
            &scene_for("engine-a", "session-a", 4, "frame-a"),
        )
        .await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if gateway.snapshot_now().is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        drop(first);
        tokio::time::sleep(Duration::from_millis(20)).await;
        let mut second = TcpStream::connect(address).await.unwrap();
        let ready = hello(&mut second, "engine-b", "session-b").await;
        assert_eq!(ready["last_transport_sequence"], Value::Null);
        assert!(gateway.snapshot_now().is_none());
        send_scene(
            &mut second,
            &scene_for("engine-b", "session-b", 1, "frame-1"),
        )
        .await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if gateway.snapshot_now().is_some_and(|state| {
                    state.current.payload.observation_status["engine_instance_id"] == "engine-b"
                }) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn lifecycle_rejects_invalid_identity_order_and_duplicate_observation() {
        let gateway = Gateway::new(test_config()).unwrap();
        let address = start_ingress(gateway.clone()).await;
        let mut producer = TcpStream::connect(address).await.unwrap();
        hello(&mut producer, "engine-a", "session-a").await;
        send_scene(
            &mut producer,
            &scene_for("wrong-engine", "session-a", 1, "frame-1"),
        )
        .await;
        let mut byte = [0];
        assert!(
            tokio::time::timeout(Duration::from_secs(1), producer.read(&mut byte))
                .await
                .unwrap()
                .is_ok()
        );
        assert!(gateway.snapshot_now().is_none());

        let mut producer = TcpStream::connect(address).await.unwrap();
        hello(&mut producer, "engine-b", "session-b").await;
        let first = scene_for("engine-b", "session-b", 1, "frame-1");
        send_scene(&mut producer, &first).await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if gateway.snapshot_now().is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let out_of_order = scene_for("engine-b", "session-b", 1, "frame-2");
        send_scene(&mut producer, &out_of_order).await;
        assert!(
            tokio::time::timeout(Duration::from_secs(1), producer.read(&mut byte))
                .await
                .unwrap()
                .is_ok_and(|read| read == 0)
        );

        let mut reconnect = TcpStream::connect(address).await.unwrap();
        let ack = hello(&mut reconnect, "engine-b", "session-b").await;
        assert_eq!(ack["last_transport_sequence"], 1);
        send_scene(
            &mut reconnect,
            &scene_for("engine-b", "session-b", 2, "frame-1"),
        )
        .await;
        assert!(
            tokio::time::timeout(Duration::from_secs(1), reconnect.read(&mut byte))
                .await
                .unwrap()
                .is_ok_and(|read| read == 0)
        );

        let mut reconnect = TcpStream::connect(address).await.unwrap();
        let ack = hello(&mut reconnect, "engine-b", "session-b").await;
        assert_eq!(ack["last_transport_sequence"], 1);
        send_scene(
            &mut reconnect,
            &scene_for("engine-b", "session-b", 2, "frame-2"),
        )
        .await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if gateway.snapshot_now().is_some_and(|state| {
                    state.current.payload.observation_status["source_frame_id"] == "frame-2"
                }) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
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
        let scene_address = start_ingress(gateway.clone()).await;
        let mut producer = TcpStream::connect(scene_address).await.unwrap();
        hello(&mut producer, "engine-a", "session-a").await;
        send_scene(
            &mut producer,
            &scene_for("engine-a", "session-a", 1, "frame-1"),
        )
        .await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if gateway.snapshot_now().is_some() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
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
        send_scene(&mut producer, &dense_scene(next_sequence)).await;
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
