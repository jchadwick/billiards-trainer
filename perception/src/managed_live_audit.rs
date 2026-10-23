//! Bounded receiver for the live detector's length-prefixed JSON timing stream.
//!
//! The socket listener only validates framing/JSON and uses a non-blocking
//! enqueue into a bounded Tokio channel. Retention and footer extraction run
//! on a separate task, so a slow audit consumer cannot back-pressure the live
//! publisher. All reported receive ticks share this receiver's monotonic
//! origin and are deliberately kept separate from producer/source timestamps.

use std::collections::{BTreeMap, VecDeque};
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, watch, Mutex, Semaphore};
use tokio::task::{JoinHandle, JoinSet};

pub const MAXIMUM_AUDIT_RECORD_BYTES: usize = 32 * 1024 * 1024;
pub const MAXIMUM_AUDIT_QUEUE_CAPACITY: usize = 65_536;
pub const MAXIMUM_RETAINED_RECORDS: usize = 1_000_000;
pub const MAXIMUM_AUDIT_CONNECTIONS: usize = 64;
pub const MAXIMUM_QUEUED_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
pub const MAXIMUM_IN_FLIGHT_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
pub const MAXIMUM_RETAINED_BYTES: usize = 512 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_AUDIT_RECORD_BYTES: usize = 1024 * 1024;
pub const DEFAULT_AUDIT_QUEUE_CAPACITY: usize = 128;
pub const DEFAULT_MAXIMUM_RETAINED_RECORDS: usize = 10_000;
pub const DEFAULT_MAXIMUM_RETAINED_BYTES: usize = 128 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_AUDIT_CONNECTIONS: usize = 4;
const OTHER_RECORD_TYPE: &str = "other";

/// Explicit memory bounds for the managed live-audit receiver.
#[derive(Debug, Clone)]
pub struct ManagedLiveAuditReceiverConfig {
    /// Must be a loopback address. Port zero asks the OS for an available port.
    pub address: SocketAddr,
    pub maximum_record_bytes: usize,
    pub queue_capacity: usize,
    pub maximum_retained_records: usize,
    pub maximum_retained_bytes: usize,
    pub maximum_connections: usize,
}

impl Default for ManagedLiveAuditReceiverConfig {
    fn default() -> Self {
        Self {
            address: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            maximum_record_bytes: DEFAULT_MAXIMUM_AUDIT_RECORD_BYTES,
            queue_capacity: DEFAULT_AUDIT_QUEUE_CAPACITY,
            maximum_retained_records: DEFAULT_MAXIMUM_RETAINED_RECORDS,
            maximum_retained_bytes: DEFAULT_MAXIMUM_RETAINED_BYTES,
            maximum_connections: DEFAULT_MAXIMUM_AUDIT_CONNECTIONS,
        }
    }
}

impl ManagedLiveAuditReceiverConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.address.ip().is_loopback(),
            "managed live-audit listener must bind to a loopback address"
        );
        ensure!(
            (1..=MAXIMUM_AUDIT_RECORD_BYTES).contains(&self.maximum_record_bytes),
            "maximum audit record bytes must be between 1 and {MAXIMUM_AUDIT_RECORD_BYTES}"
        );
        ensure!(
            (1..=MAXIMUM_AUDIT_QUEUE_CAPACITY).contains(&self.queue_capacity),
            "audit queue capacity must be between 1 and {MAXIMUM_AUDIT_QUEUE_CAPACITY}"
        );
        ensure!(
            (1..=MAXIMUM_RETAINED_RECORDS).contains(&self.maximum_retained_records),
            "maximum retained records must be between 1 and {MAXIMUM_RETAINED_RECORDS}"
        );
        ensure!(
            (1..=MAXIMUM_RETAINED_BYTES).contains(&self.maximum_retained_bytes),
            "maximum retained bytes must be between 1 and {MAXIMUM_RETAINED_BYTES}"
        );
        ensure!(
            (1..=MAXIMUM_AUDIT_CONNECTIONS).contains(&self.maximum_connections),
            "maximum audit connections must be between 1 and {MAXIMUM_AUDIT_CONNECTIONS}"
        );

        let queue_budget = self
            .queue_capacity
            .checked_mul(self.maximum_record_bytes)
            .context("audit queue memory bound overflowed")?;
        ensure!(
            queue_budget <= MAXIMUM_QUEUED_PAYLOAD_BYTES,
            "audit queue capacity and record size exceed the {MAXIMUM_QUEUED_PAYLOAD_BYTES}-byte queue budget"
        );
        let in_flight_budget = self
            .maximum_connections
            .checked_mul(self.maximum_record_bytes)
            .context("audit in-flight memory bound overflowed")?;
        ensure!(
            in_flight_budget <= MAXIMUM_IN_FLIGHT_PAYLOAD_BYTES,
            "audit connection count and record size exceed the {MAXIMUM_IN_FLIGHT_PAYLOAD_BYTES}-byte in-flight budget"
        );
        Ok(())
    }
}

/// One parsed producer record with a timestamp from the receiver's local clock.
#[derive(Debug, Clone, Serialize)]
pub struct ReceivedAuditRecord {
    /// Nanoseconds elapsed from `measurement_clock_domain_id`'s local origin.
    pub received_at_tick_ns: u64,
    /// The exact parsed producer payload. Producer clock fields are untouched.
    pub record: Value,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ManagedLiveAuditStreamCounts {
    /// Complete or malformed frame attempts with a complete four-byte header.
    pub framed_messages: u64,
    /// Valid JSON objects with a string `record_type`, including records later
    /// dropped from the audit queue.
    pub valid_records: u64,
    /// Whole-stream producer record counts, independent of detail retention.
    pub records_by_type: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ManagedLiveAuditLosses {
    pub malformed_records: u64,
    pub oversized_records: u64,
    pub queue_dropped_records: u64,
    /// Records evicted from the retained tail to satisfy its count or byte cap.
    pub retention_evictions: u64,
    /// Valid records too large to fit even in an otherwise empty retained tail.
    pub retention_oversize_drops: u64,
    pub connection_limit_drops: u64,
    pub connection_task_failures: u64,
    pub transport_errors: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ManagedLiveAuditSnapshot {
    pub listener_address: SocketAddr,
    pub measurement_clock_domain_id: String,
    /// Current receiver-local monotonic tick, in the same domain as records.
    pub measurement_clock_tick_ns: u64,
    pub stream_counts: ManagedLiveAuditStreamCounts,
    pub producer_footer: Option<Value>,
    pub losses: ManagedLiveAuditLosses,
    /// Bounded tail ordered by receipt time.
    pub retained_records: Vec<ReceivedAuditRecord>,
    pub retained_record_bytes: usize,
    pub maximum_retained_records: usize,
    pub maximum_retained_bytes: usize,
}

#[derive(Default)]
struct ReceiverState {
    stream_counts: ManagedLiveAuditStreamCounts,
    losses: ManagedLiveAuditLosses,
    producer_footer: Option<Value>,
    retained_records: VecDeque<(ReceivedAuditRecord, usize)>,
    retained_record_bytes: usize,
}

struct ReceiverShared {
    address: SocketAddr,
    clock_domain_id: String,
    clock_origin: Instant,
    maximum_retained_records: usize,
    maximum_retained_bytes: usize,
    state: Mutex<ReceiverState>,
}

struct PendingRecord {
    received_at_tick_ns: u64,
    encoded_bytes: usize,
    record: Value,
    record_type: String,
}

/// Owns the loopback listener, connection workers, and bounded retention task.
pub struct ManagedLiveAuditReceiver {
    shared: std::sync::Arc<ReceiverShared>,
    shutdown_sender: watch::Sender<bool>,
    listener_task: Option<JoinHandle<Result<()>>>,
    collector_task: Option<JoinHandle<()>>,
}

impl ManagedLiveAuditReceiver {
    /// Bind before starting the producer so the audit side-channel is ready.
    pub async fn bind(config: ManagedLiveAuditReceiverConfig) -> Result<Self> {
        config.validate()?;
        let listener = TcpListener::bind(config.address).await.with_context(|| {
            format!(
                "failed to bind managed live-audit listener at {}",
                config.address
            )
        })?;
        let address = listener.local_addr()?;
        let clock_origin = Instant::now();
        let shared = std::sync::Arc::new(ReceiverShared {
            address,
            clock_domain_id: format!("managed-live-audit-{}", uuid::Uuid::new_v4()),
            clock_origin,
            maximum_retained_records: config.maximum_retained_records,
            maximum_retained_bytes: config.maximum_retained_bytes,
            state: Mutex::new(ReceiverState::default()),
        });
        let (record_sender, record_receiver) = mpsc::channel(config.queue_capacity);
        let (shutdown_sender, shutdown_receiver) = watch::channel(false);
        let collector_shared = shared.clone();
        let collector_task = tokio::spawn(async move {
            collect_records(record_receiver, collector_shared).await;
        });
        let listener_shared = shared.clone();
        let listener_task = tokio::spawn(async move {
            run_listener(
                listener,
                listener_shared,
                record_sender,
                shutdown_receiver,
                config.maximum_record_bytes,
                config.maximum_connections,
            )
            .await
        });

        Ok(Self {
            shared,
            shutdown_sender,
            listener_task: Some(listener_task),
            collector_task: Some(collector_task),
        })
    }

    pub fn listener_address(&self) -> SocketAddr {
        self.shared.address
    }

    pub fn measurement_clock_domain_id(&self) -> &str {
        &self.shared.clock_domain_id
    }

    /// Read the local monotonic clock in the receiver's measurement domain.
    pub fn measurement_clock_tick_ns(&self) -> u64 {
        monotonic_elapsed_ns(self.shared.clock_origin)
    }

    /// Convert an `Instant` captured in this process to the receiver's local
    /// monotonic tick domain. Instants before the receiver origin saturate to
    /// tick zero; oversized durations saturate at `u64::MAX` nanoseconds.
    pub fn measurement_clock_tick_at(&self, instant: Instant) -> u64 {
        instant
            .checked_duration_since(self.shared.clock_origin)
            .map(|elapsed| u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0)
    }

    pub async fn snapshot(&self) -> ManagedLiveAuditSnapshot {
        snapshot(&self.shared).await
    }

    /// Stop accepting/reading, join connection workers, drain queued records,
    /// and return the final bounded report. No audit export work runs on the
    /// live state publication thread.
    pub async fn shutdown(mut self) -> Result<ManagedLiveAuditSnapshot> {
        let _ = self.shutdown_sender.send(true);
        let listener_result = if let Some(listener_task) = self.listener_task.take() {
            listener_task
                .await
                .context("managed live-audit listener task panicked")?
        } else {
            Ok(())
        };
        let collector_result = if let Some(collector_task) = self.collector_task.take() {
            collector_task
                .await
                .context("managed live-audit collector task panicked")
        } else {
            Ok(())
        };
        listener_result?;
        collector_result?;
        Ok(snapshot(&self.shared).await)
    }
}

impl Drop for ManagedLiveAuditReceiver {
    fn drop(&mut self) {
        // Dropping without explicit shutdown still tells tasks to exit. Callers
        // should prefer shutdown() when they need a final report and joins.
        let _ = self.shutdown_sender.send(true);
    }
}

fn monotonic_elapsed_ns(origin: Instant) -> u64 {
    u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

async fn snapshot(shared: &ReceiverShared) -> ManagedLiveAuditSnapshot {
    let state = shared.state.lock().await;
    ManagedLiveAuditSnapshot {
        listener_address: shared.address,
        measurement_clock_domain_id: shared.clock_domain_id.clone(),
        measurement_clock_tick_ns: monotonic_elapsed_ns(shared.clock_origin),
        stream_counts: state.stream_counts.clone(),
        producer_footer: state.producer_footer.clone(),
        losses: state.losses.clone(),
        retained_records: state
            .retained_records
            .iter()
            .map(|(record, _)| record.clone())
            .collect(),
        retained_record_bytes: state.retained_record_bytes,
        maximum_retained_records: shared.maximum_retained_records,
        maximum_retained_bytes: shared.maximum_retained_bytes,
    }
}

async fn run_listener(
    listener: TcpListener,
    shared: std::sync::Arc<ReceiverShared>,
    record_sender: mpsc::Sender<PendingRecord>,
    mut shutdown_receiver: watch::Receiver<bool>,
    maximum_record_bytes: usize,
    maximum_connections: usize,
) -> Result<()> {
    let permits = std::sync::Arc::new(Semaphore::new(maximum_connections));
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            changed = shutdown_receiver.changed() => {
                if changed.is_err() || *shutdown_receiver.borrow() {
                    break;
                }
            }
            completed = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = completed {
                    count_connection_task_failure(&shared, error).await;
                }
            }
            accepted = listener.accept() => {
                let (stream, _) = accepted.context("managed live-audit accept failed")?;
                // JoinSet length includes completed tasks not yet reaped, which
                // keeps both active sockets and task bookkeeping strictly bounded.
                if connections.len() >= maximum_connections {
                    let mut state = shared.state.lock().await;
                    state.losses.connection_limit_drops = state.losses.connection_limit_drops.saturating_add(1);
                    drop(stream);
                    continue;
                }
                let permit = match permits.clone().try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => {
                        let mut state = shared.state.lock().await;
                        state.losses.connection_limit_drops = state.losses.connection_limit_drops.saturating_add(1);
                        drop(stream);
                        continue;
                    }
                };
                connections.spawn(serve_connection(
                    stream,
                    shared.clone(),
                    record_sender.clone(),
                    shutdown_receiver.clone(),
                    maximum_record_bytes,
                    permit,
                ));
            }
        }
    }

    // Each connection selects on the same shutdown signal, so joining here
    // completes promptly while still dropping no queued record events.
    while let Some(result) = connections.join_next().await {
        if let Err(error) = result {
            count_connection_task_failure(&shared, error).await;
        }
    }
    drop(record_sender);
    Ok(())
}

async fn count_connection_task_failure(shared: &ReceiverShared, _error: tokio::task::JoinError) {
    let mut state = shared.state.lock().await;
    state.losses.connection_task_failures = state.losses.connection_task_failures.saturating_add(1);
}

async fn serve_connection(
    mut stream: TcpStream,
    shared: std::sync::Arc<ReceiverShared>,
    sender: mpsc::Sender<PendingRecord>,
    mut shutdown_receiver: watch::Receiver<bool>,
    maximum_record_bytes: usize,
    _permit: tokio::sync::OwnedSemaphorePermit,
) {
    loop {
        let header = tokio::select! {
            changed = shutdown_receiver.changed() => {
                if changed.is_err() || *shutdown_receiver.borrow() {
                    return;
                }
                continue;
            }
            header = read_length_header(&mut stream) => header,
        };
        let length = match header {
            HeaderRead::End => return,
            HeaderRead::Truncated => {
                let mut state = shared.state.lock().await;
                state.losses.malformed_records = state.losses.malformed_records.saturating_add(1);
                return;
            }
            HeaderRead::Length(length) => length,
            HeaderRead::IoError => {
                let mut state = shared.state.lock().await;
                state.losses.transport_errors = state.losses.transport_errors.saturating_add(1);
                return;
            }
        };
        {
            let mut state = shared.state.lock().await;
            state.stream_counts.framed_messages =
                state.stream_counts.framed_messages.saturating_add(1);
        }

        if length == 0 {
            let mut state = shared.state.lock().await;
            state.losses.malformed_records = state.losses.malformed_records.saturating_add(1);
            continue;
        }
        if length > maximum_record_bytes {
            {
                let mut state = shared.state.lock().await;
                state.losses.oversized_records = state.losses.oversized_records.saturating_add(1);
            }
            let drained = tokio::select! {
                changed = shutdown_receiver.changed() => {
                    if changed.is_err() || *shutdown_receiver.borrow() {
                        return;
                    }
                    continue;
                }
                result = discard_bytes(&mut stream, length) => result,
            };
            if !drained {
                let mut state = shared.state.lock().await;
                state.losses.malformed_records = state.losses.malformed_records.saturating_add(1);
                return;
            }
            continue;
        }

        let mut payload = Vec::new();
        if payload.try_reserve_exact(length).is_err() {
            let mut state = shared.state.lock().await;
            state.losses.malformed_records = state.losses.malformed_records.saturating_add(1);
            return;
        }
        payload.resize(length, 0);
        let payload_read = tokio::select! {
            changed = shutdown_receiver.changed() => {
                if changed.is_err() || *shutdown_receiver.borrow() {
                    return;
                }
                continue;
            }
            result = stream.read_exact(&mut payload) => result.is_ok(),
        };
        if !payload_read {
            let mut state = shared.state.lock().await;
            state.losses.malformed_records = state.losses.malformed_records.saturating_add(1);
            return;
        }

        let record: Value = match serde_json::from_slice(&payload) {
            Ok(record) => record,
            Err(_) => {
                let mut state = shared.state.lock().await;
                state.losses.malformed_records = state.losses.malformed_records.saturating_add(1);
                continue;
            }
        };
        let Some(record_type) = record
            .get("record_type")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
        else {
            let mut state = shared.state.lock().await;
            state.losses.malformed_records = state.losses.malformed_records.saturating_add(1);
            continue;
        };
        // Bound aggregate state regardless of how many distinct type strings a
        // malformed or future producer may send. Known exporter record names
        // retain their own counters; all other names share one stable bucket.
        let record_type = match record_type {
            "frame" | "scene_socket_write" | "observation_outcome" | "footer" => {
                record_type.to_owned()
            }
            _ => OTHER_RECORD_TYPE.to_owned(),
        };

        enqueue_record(&shared, &sender, record, record_type, payload.len(), None).await;
    }
}

enum HeaderRead {
    End,
    Length(usize),
    Truncated,
    IoError,
}

async fn read_length_header(stream: &mut TcpStream) -> HeaderRead {
    let mut header = [0_u8; 4];
    match stream.read(&mut header[..1]).await {
        Ok(0) => return HeaderRead::End,
        Ok(1) => {}
        Ok(_) => unreachable!("one-byte read cannot return more than one byte"),
        Err(_) => return HeaderRead::IoError,
    }
    if stream.read_exact(&mut header[1..]).await.is_err() {
        return HeaderRead::Truncated;
    }
    HeaderRead::Length(u32::from_be_bytes(header) as usize)
}

/// Drain an oversized payload in a fixed stack buffer so its length never
/// controls an allocation and subsequent framed records stay synchronized.
async fn discard_bytes(stream: &mut TcpStream, mut remaining: usize) -> bool {
    let mut scratch = [0_u8; 4096];
    while remaining > 0 {
        let count = remaining.min(scratch.len());
        if stream.read_exact(&mut scratch[..count]).await.is_err() {
            return false;
        }
        remaining -= count;
    }
    true
}

/// Update whole-stream accounting and attempt the bounded enqueue while holding
/// one short receiver-state lock. This makes the local stamp and queue order
/// consistent even if the configured listener accepts multiple producers.
async fn enqueue_record(
    shared: &ReceiverShared,
    sender: &mpsc::Sender<PendingRecord>,
    record: Value,
    record_type: String,
    encoded_bytes: usize,
    received_at_tick_ns: Option<u64>,
) {
    let mut state = shared.state.lock().await;
    state.stream_counts.valid_records = state.stream_counts.valid_records.saturating_add(1);
    let count = state
        .stream_counts
        .records_by_type
        .entry(record_type.clone())
        .or_default();
    *count = count.saturating_add(1);
    let pending = PendingRecord {
        received_at_tick_ns: received_at_tick_ns
            .unwrap_or_else(|| monotonic_elapsed_ns(shared.clock_origin)),
        encoded_bytes,
        record,
        record_type,
    };
    if sender.try_send(pending).is_err() {
        state.losses.queue_dropped_records = state.losses.queue_dropped_records.saturating_add(1);
    }
}

async fn collect_records(
    mut receiver: mpsc::Receiver<PendingRecord>,
    shared: std::sync::Arc<ReceiverShared>,
) {
    while let Some(pending) = receiver.recv().await {
        let mut state = shared.state.lock().await;
        if pending.record_type == "footer" {
            state.producer_footer = Some(pending.record.clone());
        }
        if pending.encoded_bytes > shared.maximum_retained_bytes {
            state.losses.retention_oversize_drops =
                state.losses.retention_oversize_drops.saturating_add(1);
            continue;
        }
        while !state.retained_records.is_empty()
            && (state.retained_records.len() >= shared.maximum_retained_records
                || state
                    .retained_record_bytes
                    .saturating_add(pending.encoded_bytes)
                    > shared.maximum_retained_bytes)
        {
            if let Some((_, removed_bytes)) = state.retained_records.pop_front() {
                state.retained_record_bytes =
                    state.retained_record_bytes.saturating_sub(removed_bytes);
                state.losses.retention_evictions =
                    state.losses.retention_evictions.saturating_add(1);
            }
        }
        state.retained_record_bytes = state
            .retained_record_bytes
            .saturating_add(pending.encoded_bytes);
        state.retained_records.push_back((
            ReceivedAuditRecord {
                received_at_tick_ns: pending.received_at_tick_ns,
                record: pending.record,
            },
            pending.encoded_bytes,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;

    fn config() -> ManagedLiveAuditReceiverConfig {
        ManagedLiveAuditReceiverConfig {
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            maximum_record_bytes: 1024,
            queue_capacity: 4,
            maximum_retained_records: 16,
            maximum_retained_bytes: 16 * 1024,
            maximum_connections: 2,
        }
    }

    fn framed(payload: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(4 + payload.len());
        output.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        output.extend_from_slice(payload);
        output
    }

    async fn send_bytes(address: SocketAddr, bytes: &[u8]) {
        let mut stream = TcpStream::connect(address).await.unwrap();
        stream.write_all(bytes).await.unwrap();
        stream.shutdown().await.unwrap();
    }

    async fn wait_for_records(receiver: &ManagedLiveAuditReceiver, count: u64) {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if receiver.snapshot().await.stream_counts.valid_records >= count {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("receiver did not observe expected records");
    }

    #[tokio::test]
    async fn receives_big_endian_framed_json_and_preserves_producer_clocks() {
        let receiver = ManagedLiveAuditReceiver::bind(config()).await.unwrap();
        let address = receiver.listener_address();
        let payload = br#"{"record_type":"frame","source_timestamp_ns":123456}"#;
        send_bytes(address, &framed(payload)).await;
        wait_for_records(&receiver, 1).await;

        let snapshot = receiver.shutdown().await.unwrap();
        assert_eq!(snapshot.stream_counts.framed_messages, 1);
        assert_eq!(snapshot.stream_counts.valid_records, 1);
        assert_eq!(snapshot.stream_counts.records_by_type["frame"], 1);
        assert_eq!(snapshot.retained_records.len(), 1);
        assert_eq!(
            snapshot.retained_records[0].record["source_timestamp_ns"],
            123456
        );
        assert!(
            snapshot.retained_records[0].received_at_tick_ns <= snapshot.measurement_clock_tick_ns
        );
        assert!(!snapshot.measurement_clock_domain_id.is_empty());
    }

    #[tokio::test]
    async fn malformed_and_oversized_frames_are_counted_and_stream_remains_synchronized() {
        let mut receiver_config = config();
        receiver_config.maximum_record_bytes = 64;
        let receiver = ManagedLiveAuditReceiver::bind(receiver_config)
            .await
            .unwrap();
        let address = receiver.listener_address();
        let mut stream = TcpStream::connect(address).await.unwrap();
        let mut input = Vec::new();
        input.extend_from_slice(&0_u32.to_be_bytes());
        input.extend_from_slice(&framed(b"{"));
        input.extend_from_slice(&(65_u32.to_be_bytes()));
        input.extend_from_slice(&[0_u8; 65]);
        input.extend_from_slice(&framed(br#"{"record_type":"frame"}"#));
        stream.write_all(&input).await.unwrap();
        stream.shutdown().await.unwrap();
        wait_for_records(&receiver, 1).await;

        let snapshot = receiver.shutdown().await.unwrap();
        assert_eq!(snapshot.stream_counts.framed_messages, 4);
        assert_eq!(snapshot.stream_counts.valid_records, 1);
        assert_eq!(snapshot.losses.malformed_records, 2);
        assert_eq!(snapshot.losses.oversized_records, 1);
    }

    #[tokio::test]
    async fn full_queue_drops_records_without_blocking_ingestion_and_counts_them() {
        let mut receiver_config = config();
        receiver_config.queue_capacity = 1;
        receiver_config.maximum_retained_records = 1;
        receiver_config.maximum_retained_bytes = 1024;
        let receiver = ManagedLiveAuditReceiver::bind(receiver_config)
            .await
            .unwrap();
        let (sender, mut queued) = mpsc::channel(1);
        let first = serde_json::json!({"record_type":"frame", "id":1});
        let second = serde_json::json!({"record_type":"frame", "id":2});
        submit_record(&receiver.shared, &sender, first, 3, 32).await;
        submit_record(&receiver.shared, &sender, second, 4, 32).await;
        let snapshot = receiver.snapshot().await;
        assert_eq!(snapshot.stream_counts.valid_records, 2);
        assert_eq!(snapshot.losses.queue_dropped_records, 1);
        assert_eq!(queued.try_recv().unwrap().record["id"], 1);
        drop(queued);
        drop(sender);
        let final_snapshot = receiver.shutdown().await.unwrap();
        assert_eq!(final_snapshot.losses.queue_dropped_records, 1);
    }

    #[tokio::test]
    async fn stamps_receive_times_from_one_receiver_monotonic_domain() {
        let receiver = ManagedLiveAuditReceiver::bind(config()).await.unwrap();
        assert_eq!(
            receiver.measurement_clock_tick_at(receiver.shared.clock_origin),
            0
        );
        let now_tick = receiver.measurement_clock_tick_at(Instant::now());
        assert!(now_tick <= receiver.measurement_clock_tick_ns());
        let address = receiver.listener_address();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&framed(br#"{"record_type":"frame","producer_tick":900}"#));
        bytes.extend_from_slice(&framed(br#"{"record_type":"frame","producer_tick":100}"#));
        send_bytes(address, &bytes).await;
        wait_for_records(&receiver, 2).await;

        let snapshot = receiver.shutdown().await.unwrap();
        let ticks = snapshot
            .retained_records
            .iter()
            .map(|record| record.received_at_tick_ns)
            .collect::<Vec<_>>();
        assert_eq!(ticks.len(), 2);
        assert!(ticks[0] <= ticks[1]);
        assert!(ticks[1] <= snapshot.measurement_clock_tick_ns);
        assert_eq!(snapshot.retained_records[0].record["producer_tick"], 900);
        assert_eq!(snapshot.retained_records[1].record["producer_tick"], 100);
    }

    #[tokio::test]
    async fn retention_keeps_bounded_tail_and_preserves_stream_totals_and_footer() {
        let mut receiver_config = config();
        receiver_config.maximum_retained_records = 2;
        receiver_config.maximum_retained_bytes = 4096;
        receiver_config.queue_capacity = 8;
        let receiver = ManagedLiveAuditReceiver::bind(receiver_config)
            .await
            .unwrap();
        let address = receiver.listener_address();
        let mut bytes = Vec::new();
        for id in 0..4 {
            bytes.extend_from_slice(&framed(
                serde_json::json!({"record_type":"frame", "id":id})
                    .to_string()
                    .as_bytes(),
            ));
        }
        bytes.extend_from_slice(&framed(
            br#"{"record_type":"footer","clean_shutdown":true}"#,
        ));
        send_bytes(address, &bytes).await;
        wait_for_records(&receiver, 5).await;

        let snapshot = receiver.shutdown().await.unwrap();
        assert_eq!(snapshot.stream_counts.valid_records, 5);
        assert_eq!(snapshot.stream_counts.records_by_type["frame"], 4);
        assert_eq!(snapshot.stream_counts.records_by_type["footer"], 1);
        assert_eq!(snapshot.retained_records.len(), 2);
        assert_eq!(snapshot.retained_records[0].record["id"], 3);
        assert_eq!(snapshot.retained_records[1].record["record_type"], "footer");
        assert_eq!(snapshot.producer_footer.unwrap()["clean_shutdown"], true);
        assert_eq!(snapshot.losses.retention_evictions, 3);
    }

    #[tokio::test]
    async fn shutdown_interrupts_idle_connections_and_joins_cleanly() {
        let receiver = ManagedLiveAuditReceiver::bind(config()).await.unwrap();
        let _idle_client = TcpStream::connect(receiver.listener_address())
            .await
            .unwrap();
        let snapshot = tokio::time::timeout(Duration::from_secs(2), receiver.shutdown())
            .await
            .expect("shutdown must interrupt and join idle clients")
            .unwrap();
        assert!(snapshot.retained_records.is_empty());
        assert_eq!(snapshot.stream_counts.valid_records, 0);
    }

    #[test]
    fn validation_rejects_non_loopback_addresses_and_unbounded_settings() {
        assert!(ManagedLiveAuditReceiverConfig::default().validate().is_ok());

        let mut invalid = config();
        invalid.address = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 1234);
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.queue_capacity = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.maximum_record_bytes = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.maximum_retained_bytes = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.maximum_connections = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.maximum_record_bytes = MAXIMUM_AUDIT_RECORD_BYTES;
        invalid.queue_capacity = 9;
        assert!(invalid.validate().is_err());
    }

    async fn submit_record(
        shared: &ReceiverShared,
        sender: &mpsc::Sender<PendingRecord>,
        record: Value,
        received_at_tick_ns: u64,
        encoded_bytes: usize,
    ) {
        let record_type = record["record_type"].as_str().unwrap().to_owned();
        enqueue_record(
            shared,
            sender,
            record,
            record_type,
            encoded_bytes,
            Some(received_at_tick_ns),
        )
        .await;
    }
}
