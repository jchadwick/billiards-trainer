//! Bounded newest-only scene delivery over unauthenticated loopback TCP.
//!
//! Snapshot serialization and slot replacement happen on the inference thread;
//! socket connection, protocol handshake, and writes belong to this dedicated worker.
//! The worker keeps at most one pending payload while one other payload may be
//! in flight. It never builds a historical backlog.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::capture_session_ack::CaptureSessionAcknowledgement;
use crate::live_audit::{LiveAuditRecorder, LiveAuditSceneWrite};

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
const SCENE_REPLAY_PROTOCOL_VERSION: u8 = 2;
const SCENE_LIVE_PROTOCOL_VERSION: u8 = 3;

#[derive(Debug, Clone, Copy)]
pub struct SceneTransportConfig {
    pub address: SocketAddr,
    pub maximum_message_bytes: usize,
    /// Wrap live snapshots with the Rust-owned public-state lifecycle result.
    /// Replay and offline callers keep sending raw scene-snapshot v1 records.
    pub public_state_delivery_enabled: bool,
    pub connect_timeout: Duration,
    pub write_timeout: Duration,
    pub reconnect_initial_delay: Duration,
    pub reconnect_maximum_delay: Duration,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SceneTransportStats {
    pub queued_snapshots: u64,
    /// Successful local socket writes; this is not proof of receiver acceptance.
    pub written_snapshots: u64,
    pub replaced_pending_snapshots: u64,
    /// In-flight payloads superseded while establishing a connection.
    pub superseded_before_write_snapshots: u64,
    /// Oversized payloads rejected before they enter the bounded queue.
    pub oversized_submissions: u64,
    /// Live snapshots discarded because their source-frame freshness deadline
    /// elapsed before a connection, retry, or complete socket write.
    pub expired_snapshots: u64,
    pub dropped_snapshots: u64,
    pub connection_attempts: u64,
    pub successful_connections: u64,
    pub connection_failures: u64,
    pub write_failures: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_receiver_sequence_at_handshake: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_written_sequence: Option<u64>,
}

#[derive(Default)]
struct SharedState {
    pending: Option<PendingSnapshot>,
    live_binding: Option<LiveBinding>,
    live_handshake_request: Option<LiveHandshakeRequest>,
    last_queued_sequence: Option<u64>,
    stopping: bool,
    stats: SceneTransportStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProducerIdentity {
    engine_instance_id: String,
    source_session_id: String,
}

struct PendingSnapshot {
    sequence: u64,
    identity: ProducerIdentity,
    frame_id: Option<String>,
    write_attempt: u64,
    /// Live capture snapshots are useful only until this local monotonic
    /// deadline. Offline video replay does not use this live scene transport.
    freshness_deadline: Option<Instant>,
    payload: Vec<u8>,
}

impl PendingSnapshot {
    fn is_expired(&self, now: Instant) -> bool {
        self.freshness_deadline
            .is_some_and(|deadline| now >= deadline)
    }
}

#[derive(Clone)]
struct LiveBinding {
    identity: ProducerIdentity,
    acknowledgement: CaptureSessionAcknowledgement,
}

struct LiveHandshakeRequest {
    binding: LiveBinding,
    response: Sender<std::result::Result<Option<u64>, String>>,
}

enum WorkerWork {
    Snapshot(PendingSnapshot),
    LiveHandshake(LiveHandshakeRequest),
}

/// A producer-facing latest-only queue with a dedicated network writer.
pub struct LatestSceneWriter {
    shared: Arc<(Mutex<SharedState>, Condvar)>,
    worker: Option<JoinHandle<()>>,
    maximum_message_bytes: usize,
}

impl LatestSceneWriter {
    #[cfg(test)]
    fn start(config: SceneTransportConfig) -> Result<Self> {
        Self::start_with_audit(config, None)
    }

    /// Start the newest-only scene transport with an optional independent
    /// timing side channel. The audit handle is absent in normal operation,
    /// so no audit resources are created in the worker path.
    pub(crate) fn start_with_audit(
        config: SceneTransportConfig,
        audit: Option<LiveAuditRecorder>,
    ) -> Result<Self> {
        validate_config(config)?;
        validate_handshake_size(config)?;
        let shared = Arc::new((Mutex::new(SharedState::default()), Condvar::new()));
        let worker_shared = Arc::clone(&shared);
        let worker_audit = audit.clone();
        let worker = thread::Builder::new()
            .name("scene-snapshot-writer".to_owned())
            .spawn(move || run_worker(worker_shared, config, worker_audit))
            .context("failed to start scene snapshot writer thread")?;
        Ok(Self {
            shared,
            worker: Some(worker),
            maximum_message_bytes: config.maximum_message_bytes,
        })
    }

    /// Bind one already-preflighted live source to this transport and wait for
    /// the receiver to validate its request-bound acknowledgement. This runs
    /// on the dedicated writer thread and must complete before capture reads
    /// begin.
    pub fn bind_live_session<A>(
        &self,
        engine_instance_id: &str,
        acknowledgement: A,
    ) -> Result<Option<u64>>
    where
        A: Into<CaptureSessionAcknowledgement>,
    {
        let acknowledgement = acknowledgement.into();
        validate_producer_identifier(engine_instance_id, "engine instance ID")?;
        let source_session_id = acknowledgement
            .session_id()
            .context("live acknowledgement omitted its source session ID")?;
        validate_producer_identifier(source_session_id, "source session ID")?;
        validate_live_handshake_size(self.maximum_message_bytes, &acknowledgement)?;

        let binding = LiveBinding {
            identity: ProducerIdentity {
                engine_instance_id: engine_instance_id.to_owned(),
                source_session_id: source_session_id.to_owned(),
            },
            acknowledgement,
        };
        let (response, result) = mpsc::channel();
        let (state, signal) = &*self.shared;
        {
            let mut state = lock_state(state);
            ensure!(!state.stopping, "scene snapshot writer is stopping");
            ensure!(
                state.last_queued_sequence.is_none() && state.pending.is_none(),
                "live scene session must bind before any snapshot is queued"
            );
            ensure!(
                state.live_binding.is_none() && state.live_handshake_request.is_none(),
                "scene writer already has a live session binding"
            );
            state.live_binding = Some(binding.clone());
            state.live_handshake_request = Some(LiveHandshakeRequest { binding, response });
            signal.notify_one();
        }

        result
            .recv()
            .context("scene writer stopped before completing live handshake")?
            .map_err(anyhow::Error::msg)
    }

    /// Enqueue a complete snapshot without waiting for network I/O.
    #[cfg(test)]
    fn submit(
        &self,
        sequence: u64,
        engine_instance_id: &str,
        source_session_id: &str,
        payload: Vec<u8>,
    ) -> Result<()> {
        self.submit_with_optional_frame_id(
            sequence,
            engine_instance_id,
            source_session_id,
            None,
            payload,
        )
    }

    /// Enqueue a live capture snapshot with an absolute source-frame
    /// freshness deadline. The deadline is anchored to capture read completion
    /// so submission overhead, reconnect, and socket-write time all consume
    /// the same end-to-end freshness budget.
    pub(crate) fn submit_live_with_frame_id_and_deadline(
        &self,
        sequence: u64,
        engine_instance_id: &str,
        source_session_id: &str,
        frame_id: &str,
        freshness_deadline: Instant,
        payload: Vec<u8>,
    ) -> Result<()> {
        self.submit_with_optional_frame_id_and_deadline(
            sequence,
            engine_instance_id,
            source_session_id,
            Some(frame_id),
            Some(freshness_deadline),
            payload,
        )
    }

    /// Enqueue a live phase/cue update that has no new capture-frame identity.
    /// Its independent delivery sequence still participates in the same
    /// bounded newest-only stream and expires on a local deadline.
    pub(crate) fn submit_live_event_with_deadline(
        &self,
        sequence: u64,
        engine_instance_id: &str,
        source_session_id: &str,
        freshness_deadline: Instant,
        payload: Vec<u8>,
    ) -> Result<()> {
        self.submit_with_optional_frame_id_and_deadline(
            sequence,
            engine_instance_id,
            source_session_id,
            None,
            Some(freshness_deadline),
            payload,
        )
    }

    #[cfg(test)]
    fn submit_with_optional_frame_id(
        &self,
        sequence: u64,
        engine_instance_id: &str,
        source_session_id: &str,
        frame_id: Option<&str>,
        payload: Vec<u8>,
    ) -> Result<()> {
        self.submit_with_optional_frame_id_and_deadline(
            sequence,
            engine_instance_id,
            source_session_id,
            frame_id,
            None,
            payload,
        )
    }

    fn submit_with_optional_frame_id_and_deadline(
        &self,
        sequence: u64,
        engine_instance_id: &str,
        source_session_id: &str,
        frame_id: Option<&str>,
        freshness_deadline: Option<Instant>,
        payload: Vec<u8>,
    ) -> Result<()> {
        ensure!(
            !payload.is_empty(),
            "scene snapshot payload cannot be empty"
        );
        validate_producer_identifier(engine_instance_id, "engine instance ID")?;
        validate_producer_identifier(source_session_id, "source session ID")?;
        if let Some(frame_id) = frame_id {
            validate_producer_identifier(frame_id, "source frame ID")?;
        }
        let (state, signal) = &*self.shared;
        let mut state = lock_state(state);
        ensure!(!state.stopping, "scene snapshot writer is stopping");
        ensure!(
            state
                .last_queued_sequence
                .is_none_or(|previous| sequence > previous),
            "scene transport sequence must increase for every queued snapshot"
        );
        let identity = ProducerIdentity {
            engine_instance_id: engine_instance_id.to_owned(),
            source_session_id: source_session_id.to_owned(),
        };
        if let Some(binding) = state.live_binding.as_ref() {
            ensure!(
                binding.identity == identity,
                "scene producer identity differs from the validated live session"
            );
        }
        if payload.len() > self.maximum_message_bytes {
            state.stats.oversized_submissions += 1;
            state.stats.dropped_snapshots += 1;
            anyhow::bail!(
                "scene snapshot payload is {} bytes; configured maximum is {}",
                payload.len(),
                self.maximum_message_bytes
            );
        }
        state.last_queued_sequence = Some(sequence);
        state.stats.queued_snapshots += 1;
        let pending = PendingSnapshot {
            sequence,
            identity,
            frame_id: frame_id.map(str::to_owned),
            write_attempt: 1,
            freshness_deadline,
            payload,
        };
        if state.pending.replace(pending).is_some() {
            state.stats.replaced_pending_snapshots += 1;
            state.stats.dropped_snapshots += 1;
        }
        signal.notify_one();
        Ok(())
    }

    /// Stop the worker after its current bounded attempt and return final counters.
    pub fn shutdown(&mut self) -> SceneTransportStats {
        let (state, signal) = &*self.shared;
        {
            let mut state = lock_state(state);
            state.stopping = true;
            signal.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            // Each connect/handshake and complete scene write has one absolute
            // monotonic deadline. The worker stops retrying after shutdown.
            let _ = worker.join();
        }
        self.stats()
    }

    pub fn stats(&self) -> SceneTransportStats {
        lock_state(&self.shared.0).stats.clone()
    }
}

impl Drop for LatestSceneWriter {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn validate_config(config: SceneTransportConfig) -> Result<()> {
    ensure!(
        config.address.ip().is_loopback(),
        "scene TCP address must be loopback"
    );
    ensure!(
        config.maximum_message_bytes > 0 && config.maximum_message_bytes <= u32::MAX as usize,
        "scene maximum framed message size must fit a non-zero u32"
    );
    ensure!(
        !config.connect_timeout.is_zero() && !config.write_timeout.is_zero(),
        "scene TCP connect and write timeouts must be positive"
    );
    ensure!(
        !config.reconnect_initial_delay.is_zero()
            && config.reconnect_initial_delay <= config.reconnect_maximum_delay,
        "scene reconnect delays must be positive and initial delay cannot exceed maximum"
    );
    Ok(())
}

fn validate_handshake_size(config: SceneTransportConfig) -> Result<()> {
    let hello = serde_json::json!({
        "type": "hello",
        "protocol_version": SCENE_REPLAY_PROTOCOL_VERSION,
        "engine_instance_id": "e".repeat(crate::detection_batch::MAX_IDENTIFIER_CHARACTERS),
        "source_session_id": "s".repeat(crate::detection_batch::MAX_IDENTIFIER_CHARACTERS),
    });
    let maximum_ready = serde_json::json!({
        "type": "ready",
        "protocol_version": SCENE_REPLAY_PROTOCOL_VERSION,
        "last_transport_sequence": MAX_SAFE_JSON_INTEGER,
    });
    ensure!(
        serde_json::to_vec(&hello)?.len() <= config.maximum_message_bytes
            && serde_json::to_vec(&maximum_ready)?.len() <= config.maximum_message_bytes,
        "scene maximum message size cannot carry the protocol handshake"
    );
    Ok(())
}

fn validate_live_handshake_size(
    maximum_message_bytes: usize,
    acknowledgement: &CaptureSessionAcknowledgement,
) -> Result<()> {
    let hello = serde_json::json!({
        "type": "hello",
        "protocol_version": SCENE_LIVE_PROTOCOL_VERSION,
        "engine_instance_id": "e".repeat(crate::detection_batch::MAX_IDENTIFIER_CHARACTERS),
        "source_session_id": "s".repeat(crate::detection_batch::MAX_IDENTIFIER_CHARACTERS),
        "capture_session_ack": acknowledgement,
    });
    let maximum_ready = serde_json::json!({
        "type": "ready",
        "protocol_version": SCENE_LIVE_PROTOCOL_VERSION,
        "configuration_revision": acknowledgement.configuration_revision(),
        "request_sha256": acknowledgement.request_sha256(),
        "last_transport_sequence": MAX_SAFE_JSON_INTEGER,
    });
    ensure!(
        serde_json::to_vec(&hello)?.len() <= maximum_message_bytes
            && serde_json::to_vec(&maximum_ready)?.len() <= maximum_message_bytes,
        "scene maximum message size cannot carry the live protocol handshake"
    );
    Ok(())
}

fn run_worker(
    shared: Arc<(Mutex<SharedState>, Condvar)>,
    config: SceneTransportConfig,
    audit: Option<LiveAuditRecorder>,
) {
    let mut stream: Option<TcpStream> = None;
    let mut connected_identity: Option<ProducerIdentity> = None;
    let mut receiver_sequence: Option<u64> = None;
    let mut reconnect_delay = config.reconnect_initial_delay;

    'worker: loop {
        let Some(work) = take_work(&shared) else {
            return;
        };
        if let WorkerWork::LiveHandshake(request) = work {
            increment(&shared, |stats| stats.connection_attempts += 1);
            match connect_and_handshake(
                &config,
                &request.binding.identity,
                Some(&request.binding.acknowledgement),
            ) {
                Ok((connected, accepted_sequence)) => {
                    increment(&shared, |stats| {
                        stats.successful_connections += 1;
                        stats.last_receiver_sequence_at_handshake = accepted_sequence;
                    });
                    connected_identity = Some(request.binding.identity);
                    receiver_sequence = accepted_sequence;
                    stream = Some(connected);
                    reconnect_delay = config.reconnect_initial_delay;
                    let _ = request.response.send(Ok(accepted_sequence));
                }
                Err(error) => {
                    increment(&shared, |stats| stats.connection_failures += 1);
                    {
                        let mut state = lock_state(&shared.0);
                        if state
                            .live_binding
                            .as_ref()
                            .is_some_and(|binding| binding.identity == request.binding.identity)
                        {
                            state.live_binding = None;
                        }
                    }
                    let _ = request.response.send(Err(format!("{error:#}")));
                }
            }
            continue;
        }
        let WorkerWork::Snapshot(mut pending) = work else {
            unreachable!("live handshake requests return above")
        };
        let stopping = lock_state(&shared.0).stopping;

        if drop_if_expired(&shared, &pending) {
            continue 'worker;
        }

        loop {
            if stream.is_none() || connected_identity.as_ref() != Some(&pending.identity) {
                stream = None;
                receiver_sequence = None;
                increment(&shared, |stats| stats.connection_attempts += 1);
                let acknowledgement = live_acknowledgement_for(&shared, &pending.identity);
                match connect_and_handshake(&config, &pending.identity, acknowledgement.as_ref()) {
                    Ok((connected, accepted_sequence)) => {
                        receiver_sequence = accepted_sequence;
                        increment(&shared, |stats| {
                            stats.successful_connections += 1;
                            stats.last_receiver_sequence_at_handshake = accepted_sequence;
                        });
                        connected_identity = Some(pending.identity.clone());
                        stream = Some(connected);
                        reconnect_delay = config.reconnect_initial_delay;
                        if drop_if_expired(&shared, &pending) {
                            continue 'worker;
                        }
                    }
                    Err(_) => {
                        increment(&shared, |stats| stats.connection_failures += 1);
                        if stopping {
                            count_shutdown_drop(&shared, true);
                            return;
                        }
                        requeue_or_drop(&shared, pending);
                        if wait_to_retry(&shared, reconnect_delay) {
                            count_shutdown_drop(&shared, false);
                            return;
                        }
                        reconnect_delay =
                            doubled_delay(reconnect_delay, config.reconnect_maximum_delay);
                        continue 'worker;
                    }
                }
            }

            pending = supersede_with_newest_pending(&shared, pending);
            if drop_if_expired(&shared, &pending) {
                continue 'worker;
            }
            if connected_identity.as_ref() == Some(&pending.identity) {
                break;
            }
            // A source session or engine restart arrived during handshake.
            // Re-handshake with that epoch before any of its scenes are sent.
            stream = None;
            receiver_sequence = None;
        }

        if receiver_sequence.is_some_and(|accepted| pending.sequence <= accepted) {
            // The receiver's handshake watermark proves it already admitted
            // this sequence for this exact producer epoch.
            continue;
        }

        // A pending capture may expire while the receiver handshake or
        // watermark check is in progress. Check again immediately before the
        // first byte can be written.
        if drop_if_expired(&shared, &pending) {
            continue 'worker;
        }
        let write_started = Instant::now();
        let write_result = match Instant::now().checked_add(config.write_timeout) {
            Some(write_timeout_deadline) => {
                let deadline = pending
                    .freshness_deadline
                    .map_or(write_timeout_deadline, |freshness_deadline| {
                        freshness_deadline.min(write_timeout_deadline)
                    });
                write_framed_message_until(
                    stream
                        .as_mut()
                        .expect("connected stream was initialized above"),
                    &pending.payload,
                    config.maximum_message_bytes,
                    deadline,
                )
            }
            None => Err(anyhow::anyhow!(
                "scene write deadline cannot be represented"
            )),
        };
        if let (Some(audit), Some(frame_id)) = (audit.as_ref(), pending.frame_id.as_deref()) {
            audit.record_scene_socket_write(LiveAuditSceneWrite {
                engine_instance_id: &pending.identity.engine_instance_id,
                source_session_id: &pending.identity.source_session_id,
                frame_id,
                transport_sequence: pending.sequence,
                write_attempt: pending.write_attempt,
                duration_ms: write_started.elapsed().as_secs_f64() * 1_000.0,
                succeeded: write_result.is_ok(),
            });
        }
        match write_result {
            Ok(()) => {
                receiver_sequence = Some(pending.sequence);
                increment(&shared, |stats| {
                    stats.written_snapshots += 1;
                    stats.last_written_sequence = Some(pending.sequence);
                });
            }
            Err(_) => {
                stream = None;
                connected_identity = None;
                receiver_sequence = None;
                if drop_if_expired(&shared, &pending) {
                    continue 'worker;
                }
                increment(&shared, |stats| stats.write_failures += 1);
                if stopping {
                    count_shutdown_drop(&shared, true);
                    return;
                }
                pending.write_attempt = pending.write_attempt.saturating_add(1);
                requeue_or_drop(&shared, pending);
                if wait_to_retry(&shared, reconnect_delay) {
                    count_shutdown_drop(&shared, false);
                    return;
                }
                reconnect_delay = doubled_delay(reconnect_delay, config.reconnect_maximum_delay);
            }
        }
    }
}

fn connect_and_handshake(
    config: &SceneTransportConfig,
    identity: &ProducerIdentity,
    acknowledgement: Option<&CaptureSessionAcknowledgement>,
) -> Result<(TcpStream, Option<u64>)> {
    let deadline = Instant::now()
        .checked_add(config.connect_timeout)
        .context("scene connection deadline cannot be represented")?;
    let mut stream = TcpStream::connect_timeout(&config.address, config.connect_timeout)
        .with_context(|| format!("failed to connect to scene receiver at {}", config.address))?;
    stream
        .set_nodelay(true)
        .context("failed to enable low-latency scene TCP writes")?;

    let hello = if let Some(acknowledgement) = acknowledgement {
        ensure!(
            acknowledgement.session_id() == Some(identity.source_session_id.as_str()),
            "live acknowledgement session ID differs from producer identity"
        );
        serde_json::json!({
            "type": "hello",
            "protocol_version": SCENE_LIVE_PROTOCOL_VERSION,
            "engine_instance_id": identity.engine_instance_id,
            "source_session_id": identity.source_session_id,
            "capture_session_ack": acknowledgement,
        })
    } else {
        serde_json::json!({
            "type": "hello",
            "protocol_version": SCENE_REPLAY_PROTOCOL_VERSION,
            "engine_instance_id": identity.engine_instance_id,
            "source_session_id": identity.source_session_id,
        })
    };
    let hello_bytes = serde_json::to_vec(&hello).context("failed to serialize scene hello")?;
    write_framed_message_until(
        &mut stream,
        &hello_bytes,
        config.maximum_message_bytes,
        deadline,
    )
    .context("failed to send scene protocol hello")?;

    let mut header = [0_u8; 4];
    read_exact_until(&mut stream, &mut header, deadline)
        .context("scene receiver did not return its ready response")?;
    let response_size = u32::from_be_bytes(header) as usize;
    ensure!(
        response_size > 0 && response_size <= config.maximum_message_bytes,
        "scene receiver ready response has an invalid size"
    );
    let mut response_bytes = vec![0_u8; response_size];
    read_exact_until(&mut stream, &mut response_bytes, deadline)
        .context("scene receiver ready response was truncated")?;
    let response: Value = serde_json::from_slice(&response_bytes)
        .context("scene receiver ready response is not valid JSON")?;
    if let Some(acknowledgement) = acknowledgement {
        let expected_keys = [
            "type",
            "protocol_version",
            "configuration_revision",
            "request_sha256",
            "last_transport_sequence",
        ];
        ensure!(
            response.as_object().is_some_and(|object| {
                object.len() == expected_keys.len()
                    && expected_keys.iter().all(|key| object.contains_key(*key))
            }) && response.get("type").and_then(Value::as_str) == Some("ready")
                && response.get("protocol_version").and_then(Value::as_u64)
                    == Some(u64::from(SCENE_LIVE_PROTOCOL_VERSION))
                && response
                    .get("configuration_revision")
                    .and_then(Value::as_u64)
                    == Some(acknowledgement.configuration_revision())
                && response.get("request_sha256").and_then(Value::as_str)
                    == Some(acknowledgement.request_sha256()),
            "scene receiver did not acknowledge the matching live protocol request"
        );
    } else {
        ensure!(
            response.get("type").and_then(Value::as_str) == Some("ready")
                && response.get("protocol_version").and_then(Value::as_u64)
                    == Some(u64::from(SCENE_REPLAY_PROTOCOL_VERSION)),
            "scene receiver did not acknowledge protocol version 2"
        );
    }
    let last_sequence = match response.get("last_transport_sequence") {
        Some(Value::Null) => None,
        Some(value) => {
            let sequence = value
                .as_u64()
                .context("scene receiver returned an invalid sequence watermark")?;
            ensure!(
                sequence <= MAX_SAFE_JSON_INTEGER,
                "scene receiver returned an unsafe sequence watermark"
            );
            Some(sequence)
        }
        None => anyhow::bail!("scene receiver omitted its sequence watermark"),
    };
    Ok((stream, last_sequence))
}

/// Send a complete frame within one monotonic deadline. `write_all` alone is
/// insufficient here: its repeated partial writes can each receive a fresh
/// socket timeout when the peer slowly drains data.
fn write_framed_message_until(
    stream: &mut TcpStream,
    payload: &[u8],
    maximum_message_bytes: usize,
    deadline: Instant,
) -> Result<()> {
    ensure!(
        maximum_message_bytes > 0,
        "scene message limit must be positive"
    );
    ensure!(!payload.is_empty(), "scene frame cannot be empty");
    ensure!(
        payload.len() <= maximum_message_bytes && payload.len() <= u32::MAX as usize,
        "scene frame exceeds configured maximum"
    );
    let header = (payload.len() as u32).to_be_bytes();
    write_all_until(stream, &header, deadline).context("failed to write scene frame header")?;
    write_all_until(stream, payload, deadline).context("failed to write scene frame body")?;
    Ok(())
}

fn write_all_until(stream: &mut TcpStream, mut bytes: &[u8], deadline: Instant) -> io::Result<()> {
    while !bytes.is_empty() {
        let remaining = remaining_before(deadline)?;
        stream.set_write_timeout(Some(remaining))?;
        match stream.write(bytes) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write the complete scene frame",
                ));
            }
            Ok(written) => bytes = &bytes[written..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn read_exact_until(
    stream: &mut TcpStream,
    mut bytes: &mut [u8],
    deadline: Instant,
) -> io::Result<()> {
    while !bytes.is_empty() {
        let remaining = remaining_before(deadline)?;
        stream.set_read_timeout(Some(remaining))?;
        match stream.read(bytes) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer closed before the complete scene response arrived",
                ));
            }
            Ok(read) => bytes = &mut bytes[read..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn remaining_before(deadline: Instant) -> io::Result<Duration> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "scene socket operation exceeded its total deadline",
        ));
    }
    Ok(remaining)
}

fn take_work(shared: &Arc<(Mutex<SharedState>, Condvar)>) -> Option<WorkerWork> {
    let (state, signal) = &**shared;
    let mut state = lock_state(state);
    loop {
        if let Some(request) = state.live_handshake_request.take() {
            return Some(WorkerWork::LiveHandshake(request));
        }
        if let Some(pending) = state.pending.take() {
            return Some(WorkerWork::Snapshot(pending));
        }
        if state.stopping {
            return None;
        }
        state = signal
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn live_acknowledgement_for(
    shared: &Arc<(Mutex<SharedState>, Condvar)>,
    identity: &ProducerIdentity,
) -> Option<CaptureSessionAcknowledgement> {
    lock_state(&shared.0)
        .live_binding
        .as_ref()
        .filter(|binding| &binding.identity == identity)
        .map(|binding| binding.acknowledgement.clone())
}

/// Replace a snapshot that has not started writing with any newer pending
/// snapshot. In particular, a slow handshake must not put an already stale
/// location ahead of the freshest scene waiting in the one-item slot.
fn supersede_with_newest_pending(
    shared: &Arc<(Mutex<SharedState>, Condvar)>,
    pending: PendingSnapshot,
) -> PendingSnapshot {
    let mut state = lock_state(&shared.0);
    if let Some(newest) = state.pending.take() {
        state.stats.superseded_before_write_snapshots += 1;
        state.stats.dropped_snapshots += 1;
        newest
    } else {
        pending
    }
}

/// Expire an in-flight live capture snapshot exactly once before it can be
/// retried or written. Returning true means the caller should abandon it and
/// take the newest work item, if any.
fn drop_if_expired(shared: &Arc<(Mutex<SharedState>, Condvar)>, pending: &PendingSnapshot) -> bool {
    if !pending.is_expired(Instant::now()) {
        return false;
    }
    let mut state = lock_state(&shared.0);
    state.stats.expired_snapshots += 1;
    state.stats.dropped_snapshots += 1;
    true
}

fn requeue_or_drop(shared: &Arc<(Mutex<SharedState>, Condvar)>, pending: PendingSnapshot) {
    let mut state = lock_state(&shared.0);
    if pending.is_expired(Instant::now()) {
        state.stats.expired_snapshots += 1;
        state.stats.dropped_snapshots += 1;
    } else if state.pending.is_some() {
        state.stats.dropped_snapshots += 1;
    } else {
        state.pending = Some(pending);
    }
    shared.1.notify_one();
}

fn count_shutdown_drop(shared: &Arc<(Mutex<SharedState>, Condvar)>, in_flight: bool) {
    let mut state = lock_state(&shared.0);
    if in_flight {
        state.stats.dropped_snapshots += 1;
    }
    if state.pending.take().is_some() {
        state.stats.dropped_snapshots += 1;
    }
}

fn wait_to_retry(shared: &Arc<(Mutex<SharedState>, Condvar)>, delay: Duration) -> bool {
    let state = lock_state(&shared.0);
    if state.stopping {
        return true;
    }
    let deadline = Instant::now() + delay;
    let mut state = state;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return state.stopping;
        }
        let (next_state, timeout) = shared
            .1
            .wait_timeout(state, remaining)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next_state;
        if state.stopping || timeout.timed_out() {
            return state.stopping;
        }
        // Submitting a newer snapshot must not bypass the configured retry
        // backoff when the receiver is unavailable.
    }
}

fn increment(
    shared: &Arc<(Mutex<SharedState>, Condvar)>,
    update: impl FnOnce(&mut SceneTransportStats),
) {
    update(&mut lock_state(&shared.0).stats);
}

fn lock_state(mutex: &Mutex<SharedState>) -> MutexGuard<'_, SharedState> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn doubled_delay(current: Duration, maximum: Duration) -> Duration {
    current.checked_mul(2).unwrap_or(maximum).min(maximum)
}

fn validate_producer_identifier(identifier: &str, name: &str) -> Result<()> {
    ensure!(
        !identifier.is_empty()
            && identifier.chars().count() <= crate::detection_batch::MAX_IDENTIFIER_CHARACTERS
            && !identifier.chars().any(char::is_control),
        "scene producer {name} must be a bounded non-empty identifier"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use serde_json::{json, Value};

    use super::{connect_and_handshake, LatestSceneWriter, ProducerIdentity, SceneTransportConfig};

    fn config(address: std::net::SocketAddr) -> SceneTransportConfig {
        SceneTransportConfig {
            address,
            maximum_message_bytes: 64 * 1024,
            public_state_delivery_enabled: false,
            connect_timeout: Duration::from_millis(500),
            write_timeout: Duration::from_millis(500),
            reconnect_initial_delay: Duration::from_millis(10),
            reconnect_maximum_delay: Duration::from_millis(50),
        }
    }

    #[test]
    fn rejects_a_message_limit_that_cannot_carry_the_ready_watermark() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut transport = config(address);
        transport.maximum_message_bytes = 60;
        assert!(LatestSceneWriter::start(transport).is_err());
    }

    #[test]
    fn rejects_oversized_snapshots_before_they_enter_the_queue() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut transport = config(address);
        transport.maximum_message_bytes = 512;
        let mut writer = LatestSceneWriter::start(transport).unwrap();

        let error = writer
            .submit(1, "engine-a", "session-a", vec![b'x'; 513])
            .unwrap_err();
        assert!(error.to_string().contains("configured maximum is 512"));
        let stats = writer.stats();
        assert_eq!(stats.queued_snapshots, 0);
        assert_eq!(stats.oversized_submissions, 1);
        assert_eq!(stats.dropped_snapshots, 1);
        assert!(writer
            .submit(1, "engine-a", "session-a", b"{}".to_vec())
            .is_ok());
        let final_stats = writer.shutdown();
        assert_eq!(final_stats.queued_snapshots, 1);
    }

    #[test]
    fn handshake_deadline_is_total_even_when_peer_trickles_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let peer = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame(&mut stream);
            let header = (64_u32).to_be_bytes();
            for byte in header {
                if stream.write_all(&[byte]).is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(120));
            }
        });

        let mut transport = config(address);
        transport.connect_timeout = Duration::from_millis(300);
        let start = Instant::now();
        let identity = ProducerIdentity {
            engine_instance_id: "engine-a".to_owned(),
            source_session_id: "session-a".to_owned(),
        };
        let error = connect_and_handshake(&transport, &identity, None).unwrap_err();
        let elapsed = start.elapsed();
        peer.join().unwrap();

        assert!(error.to_string().contains("ready response"));
        assert!(elapsed < Duration::from_millis(500), "elapsed: {elapsed:?}");
    }

    #[test]
    fn shutdown_is_bounded_when_a_peer_slowly_drains_a_large_scene() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame(&mut stream);
            write_ready(&mut stream, None);
            ready_tx.send(()).unwrap();

            let deadline = Instant::now() + Duration::from_millis(700);
            let mut bytes = [0_u8; 1024];
            while Instant::now() < deadline {
                match stream.read(&mut bytes) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => thread::sleep(Duration::from_millis(20)),
                }
            }
        });

        let mut transport = config(address);
        transport.maximum_message_bytes = 16 * 1024 * 1024;
        transport.write_timeout = Duration::from_millis(150);
        let mut writer = LatestSceneWriter::start(transport).unwrap();
        writer
            .submit(1, "engine-a", "session-a", vec![b'x'; 16 * 1024 * 1024])
            .unwrap();
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let start = Instant::now();
        let stats = writer.shutdown();
        let elapsed = start.elapsed();
        receiver.join().unwrap();

        assert!(elapsed < Duration::from_millis(600), "elapsed: {elapsed:?}");
        assert!(stats.write_failures > 0);
        assert!(stats.dropped_snapshots > 0);
    }

    fn read_frame(stream: &mut impl Read) -> Vec<u8> {
        let mut header = [0_u8; 4];
        stream.read_exact(&mut header).unwrap();
        let size = u32::from_be_bytes(header) as usize;
        let mut payload = vec![0_u8; size];
        stream.read_exact(&mut payload).unwrap();
        payload
    }

    fn write_ready(stream: &mut impl Write, last_sequence: Option<u64>) {
        let ready = serde_json::to_vec(&json!({
            "type": "ready",
            "protocol_version": 2,
            "last_transport_sequence": last_sequence,
        }))
        .unwrap();
        crate::framing::write_framed_message(stream, &ready, 64 * 1024).unwrap();
    }

    fn write_live_ready(
        stream: &mut impl Write,
        acknowledgement: &crate::capture_session_ack::CaptureSessionAcknowledgementV1,
        last_sequence: Option<u64>,
    ) {
        let ready = serde_json::to_vec(&json!({
            "type": "ready",
            "protocol_version": 3,
            "configuration_revision": acknowledgement.configuration_revision(),
            "request_sha256": acknowledgement.request_sha256(),
            "last_transport_sequence": last_sequence,
        }))
        .unwrap();
        crate::framing::write_framed_message(stream, &ready, 64 * 1024).unwrap();
    }

    fn wait_until_stats(
        writer: &LatestSceneWriter,
        timeout: Duration,
        predicate: impl Fn(&super::SceneTransportStats) -> bool,
    ) {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if predicate(&writer.stats()) {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!(
            "scene transport stats condition was not reached: {:?}",
            writer.stats()
        );
    }

    #[test]
    fn live_binding_waits_for_request_bound_ready_before_any_scene_is_submitted() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let acknowledgement = crate::capture_session_ack::default_test_acknowledgement();
        let expected_acknowledgement = serde_json::to_value(&acknowledgement).unwrap();
        let session_id = acknowledgement.session_id().unwrap().to_owned();
        let (hello_tx, hello_rx) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let hello: Value = serde_json::from_slice(&read_frame(&mut stream)).unwrap();
            assert_eq!(
                hello,
                json!({
                    "type": "hello",
                    "protocol_version": 3,
                    "engine_instance_id": "live-engine",
                    "source_session_id": session_id,
                    "capture_session_ack": expected_acknowledgement,
                })
            );
            write_live_ready(
                &mut stream,
                &crate::capture_session_ack::default_test_acknowledgement(),
                None,
            );
            hello_tx.send(()).unwrap();
            let snapshot: Value = serde_json::from_slice(&read_frame(&mut stream)).unwrap();
            snapshot["transport_sequence"].as_u64().unwrap()
        });

        let mut writer = LatestSceneWriter::start(config(address)).unwrap();
        let accepted_sequence = writer
            .bind_live_session(
                "live-engine",
                crate::capture_session_ack::default_test_acknowledgement(),
            )
            .unwrap();
        assert_eq!(accepted_sequence, None);
        hello_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(writer
            .submit(1, "live-engine", "wrong-source-session", b"{}".to_vec())
            .is_err());
        writer
            .submit(
                1,
                "live-engine",
                "capture-session-20261020-1",
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .unwrap();
        assert_eq!(writer.shutdown().written_snapshots, 1);
        assert_eq!(receiver.join().unwrap(), 1);
    }

    #[test]
    fn live_binding_rejects_a_ready_response_for_another_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _hello = read_frame(&mut stream);
            crate::framing::write_framed_message(
                &mut stream,
                &serde_json::to_vec(&json!({
                    "type": "ready",
                    "protocol_version": 3,
                    "configuration_revision": 1,
                    "request_sha256": "0".repeat(64),
                    "last_transport_sequence": null,
                }))
                .unwrap(),
                64 * 1024,
            )
            .unwrap();
            ready_tx.send(()).unwrap();
        });

        let mut writer = LatestSceneWriter::start(config(address)).unwrap();
        let error = writer
            .bind_live_session(
                "live-engine",
                crate::capture_session_ack::default_test_acknowledgement(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("matching live protocol request"));
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let stats = writer.shutdown();
        receiver.join().unwrap();
        assert_eq!(stats.successful_connections, 0);
        assert_eq!(stats.connection_failures, 1);
    }

    #[test]
    fn coalesces_pending_work_and_does_not_block_the_submitter_on_handshake() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let hello: Value = serde_json::from_slice(&read_frame(&mut stream)).unwrap();
            assert_eq!(
                hello,
                json!({
                    "type": "hello",
                    "protocol_version": 2,
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                })
            );
            ready_tx.send(()).unwrap();
            thread::sleep(Duration::from_millis(80));
            write_ready(&mut stream, None);
            let latest: Value = serde_json::from_slice(&read_frame(&mut stream)).unwrap();
            latest["transport_sequence"].as_u64().unwrap()
        });

        let mut writer = LatestSceneWriter::start(config(address)).unwrap();
        writer
            .submit(
                1,
                "engine-a",
                "session-a",
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .unwrap();
        assert!(writer
            .submit(
                1,
                "engine-a",
                "session-a",
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .is_err());
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        for sequence in 2..=3 {
            writer
                .submit(
                    sequence,
                    "engine-a",
                    "session-a",
                    serde_json::to_vec(&json!({"transport_sequence": sequence})).unwrap(),
                )
                .unwrap();
        }
        let start = Instant::now();
        let stats = writer.shutdown();
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(receiver.join().unwrap(), 3);
        assert_eq!(stats.queued_snapshots, 3);
        assert_eq!(stats.replaced_pending_snapshots, 1);
        assert_eq!(stats.superseded_before_write_snapshots, 1);
        assert_eq!(stats.dropped_snapshots, 2);
        assert_eq!(stats.written_snapshots, 1);
    }

    #[test]
    fn slow_reader_does_not_turn_the_producer_queue_into_a_backlog() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame(&mut stream);
            write_ready(&mut stream, None);
            ready_tx.send(()).unwrap();
            // Deliberately hold the connection open without reading snapshots.
            thread::sleep(Duration::from_millis(350));
        });

        let mut transport = config(address);
        transport.write_timeout = Duration::from_millis(100);
        let mut writer = LatestSceneWriter::start(transport).unwrap();
        writer
            .submit(
                1,
                "engine-a",
                "session-a",
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .unwrap();
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let payload = vec![b'x'; 64 * 1024];
        let submit_start = Instant::now();
        for sequence in 2..=1500 {
            writer
                .submit(sequence, "engine-a", "session-a", payload.clone())
                .unwrap();
        }
        let submit_elapsed = submit_start.elapsed();
        let stats = writer.shutdown();
        receiver.join().unwrap();

        assert!(submit_elapsed < Duration::from_secs(3));
        assert_eq!(stats.queued_snapshots, 1500);
        assert!(stats.replaced_pending_snapshots > 0);
        assert!(stats.dropped_snapshots > 0);
        // Kernel socket-buffer sizes differ by OS and architecture. This test
        // verifies latest-only queue coalescing; the large-frame shutdown test
        // above separately forces write backpressure and verifies its deadline.
    }

    #[test]
    fn reconnect_uses_receiver_watermark_and_delivers_latest_sequence() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (first_sent_tx, first_sent_rx) = mpsc::channel();
        let (allow_reconnect_tx, allow_reconnect_rx) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let _ = read_frame(&mut first);
            write_ready(&mut first, None);
            let first_scene: Value = serde_json::from_slice(&read_frame(&mut first)).unwrap();
            assert_eq!(first_scene["transport_sequence"], 1);
            first_sent_tx.send(()).unwrap();
            drop(first);
            allow_reconnect_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap();

            let (mut second, _) = listener.accept().unwrap();
            let _ = read_frame(&mut second);
            write_ready(&mut second, Some(1));
            let second_scene: Value = serde_json::from_slice(&read_frame(&mut second)).unwrap();
            second_scene["transport_sequence"].as_u64().unwrap()
        });

        let mut writer = LatestSceneWriter::start(config(address)).unwrap();
        writer
            .submit(
                1,
                "engine-a",
                "session-a",
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .unwrap();
        first_sent_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        allow_reconnect_tx.send(()).unwrap();
        let mut final_submitted = 2;
        let reconnect_deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < reconnect_deadline && writer.stats().successful_connections < 2 {
            writer
                .submit(
                    final_submitted,
                    "engine-a",
                    "session-a",
                    serde_json::to_vec(&json!({"transport_sequence": final_submitted})).unwrap(),
                )
                .unwrap();
            thread::sleep(Duration::from_millis(20));
            final_submitted += 1;
        }
        assert!(writer.stats().successful_connections >= 2);
        let stats = writer.shutdown();
        assert!(receiver.join().unwrap() >= 2);
        assert!(stats.successful_connections >= 2);
        assert_eq!(stats.last_receiver_sequence_at_handshake, Some(1));
        assert!(stats.last_written_sequence.unwrap() >= 2);
    }

    #[test]
    fn expired_live_snapshot_is_not_replayed_after_receiver_returns() {
        let reserved = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reserved.local_addr().unwrap();
        drop(reserved);

        let mut transport = config(address);
        transport.reconnect_initial_delay = Duration::from_millis(20);
        transport.reconnect_maximum_delay = Duration::from_millis(40);
        let mut writer = LatestSceneWriter::start(transport).unwrap();
        let freshness_deadline = Instant::now() + Duration::from_millis(120);
        writer
            .submit_live_with_frame_id_and_deadline(
                1,
                "live-engine",
                "live-session",
                "frame-1",
                freshness_deadline,
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .unwrap();

        wait_until_stats(&writer, Duration::from_secs(1), |stats| {
            stats.connection_failures > 0
        });
        wait_until_stats(&writer, Duration::from_secs(1), |stats| {
            stats.expired_snapshots == 1
        });

        // The receiver becomes available only after the source frame's full
        // freshness budget has elapsed. No handshake or stale scene should
        // occur until a new capture observation is submitted.
        let receiver = TcpListener::bind(address).unwrap();
        receiver.set_nonblocking(true).unwrap();
        thread::sleep(Duration::from_millis(60));
        assert!(receiver.accept().is_err());
        let stats = writer.shutdown();
        assert_eq!(stats.expired_snapshots, 1);
        assert_eq!(stats.written_snapshots, 0);
        assert_eq!(stats.dropped_snapshots, 1);
    }

    #[test]
    fn reconnect_retries_only_the_newest_live_snapshot() {
        let reserved = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reserved.local_addr().unwrap();
        drop(reserved);

        let (received_tx, received_rx) = mpsc::channel();
        let mut transport = config(address);
        transport.reconnect_initial_delay = Duration::from_millis(300);
        transport.reconnect_maximum_delay = Duration::from_millis(300);
        let mut writer = LatestSceneWriter::start(transport).unwrap();
        let first_deadline = Instant::now() + Duration::from_secs(3);
        writer
            .submit_live_with_frame_id_and_deadline(
                1,
                "live-engine",
                "live-session",
                "frame-1",
                first_deadline,
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .unwrap();
        wait_until_stats(&writer, Duration::from_secs(1), |stats| {
            stats.connection_failures > 0
        });

        for sequence in 2..=3 {
            let freshness_deadline = Instant::now() + Duration::from_secs(3);
            writer
                .submit_live_with_frame_id_and_deadline(
                    sequence,
                    "live-engine",
                    "live-session",
                    &format!("frame-{sequence}"),
                    freshness_deadline,
                    serde_json::to_vec(&json!({"transport_sequence": sequence})).unwrap(),
                )
                .unwrap();
        }

        let receiver = thread::spawn(move || {
            let listener = TcpListener::bind(address).unwrap();
            let (mut stream, _) = listener.accept().unwrap();
            let _hello = read_frame(&mut stream);
            write_ready(&mut stream, None);
            let received: Value = serde_json::from_slice(&read_frame(&mut stream)).unwrap();
            received_tx
                .send(received["transport_sequence"].as_u64().unwrap())
                .unwrap();
        });

        assert_eq!(received_rx.recv_timeout(Duration::from_secs(2)).unwrap(), 3);
        let stats = writer.shutdown();
        receiver.join().unwrap();
        assert_eq!(stats.last_written_sequence, Some(3));
        assert!(stats.replaced_pending_snapshots >= 1);
        assert_eq!(stats.expired_snapshots, 0);
    }

    #[test]
    fn producer_epoch_change_reconnects_and_requests_a_fresh_watermark() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (first_sent_tx, first_sent_rx) = mpsc::channel();
        let receiver = thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let first_hello: Value = serde_json::from_slice(&read_frame(&mut first)).unwrap();
            assert_eq!(first_hello["engine_instance_id"], "engine-a");
            assert_eq!(first_hello["source_session_id"], "session-a");
            write_ready(&mut first, None);
            let first_scene: Value = serde_json::from_slice(&read_frame(&mut first)).unwrap();
            assert_eq!(first_scene["transport_sequence"], 1);
            first_sent_tx.send(()).unwrap();
            drop(first);

            let (mut restarted, _) = listener.accept().unwrap();
            let restarted_hello: Value =
                serde_json::from_slice(&read_frame(&mut restarted)).unwrap();
            assert_eq!(restarted_hello["engine_instance_id"], "engine-b");
            assert_eq!(restarted_hello["source_session_id"], "session-b");
            // A new identity epoch has no prior sequence watermark even if the
            // previous producer's receiver watermark was higher.
            write_ready(&mut restarted, None);
            let restarted_scene: Value =
                serde_json::from_slice(&read_frame(&mut restarted)).unwrap();
            restarted_scene["transport_sequence"].as_u64().unwrap()
        });

        let mut writer = LatestSceneWriter::start(config(address)).unwrap();
        writer
            .submit(
                1,
                "engine-a",
                "session-a",
                serde_json::to_vec(&json!({"transport_sequence": 1})).unwrap(),
            )
            .unwrap();
        first_sent_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        writer
            .submit(
                2,
                "engine-b",
                "session-b",
                serde_json::to_vec(&json!({"transport_sequence": 2})).unwrap(),
            )
            .unwrap();

        assert_eq!(receiver.join().unwrap(), 2);
        let stats = writer.shutdown();
        assert_eq!(stats.successful_connections, 2);
        assert_eq!(stats.last_receiver_sequence_at_handshake, None);
    }

    #[test]
    fn rejects_a_receiver_watermark_above_the_json_safe_integer_range() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = read_frame(&mut stream);
            let ready = serde_json::to_vec(&json!({
                "type": "ready",
                "protocol_version": 2,
                "last_transport_sequence": 9_007_199_254_740_992_u64,
            }))
            .unwrap();
            crate::framing::write_framed_message(&mut stream, &ready, 64 * 1024).unwrap();
        });

        let identity = ProducerIdentity {
            engine_instance_id: "engine-a".to_owned(),
            source_session_id: "session-a".to_owned(),
        };
        let error = connect_and_handshake(&config(address), &identity, None).unwrap_err();
        receiver.join().unwrap();
        assert!(error.to_string().contains("unsafe sequence watermark"));
    }
}
