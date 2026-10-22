//! Bounded loopback receiver used to qualify detection-batch replay output.
//!
//! This is an opt-in development tool, not part of the live server. It accepts
//! one producer, validates every framed batch against the shared Rust contract,
//! records receiver-local monotonic receipts, and preserves each valid payload
//! as JSONL for the retained cross-runtime scorer.

use std::collections::HashSet;
use std::io;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::timeout;
use uuid::Uuid;

use crate::detection_batch::{decode_detection_batch_v1, DetectionBatchV1};

const RECEIVER_REPORT_SCHEMA_VERSION: u8 = 1;
const NANOS_PER_MILLISECOND: f64 = 1_000_000.0;
const SHA256_HEX_CHARACTERS: usize = 64;

/// Bounds and time limits for one receiver invocation.
#[derive(Debug, Clone)]
pub struct ReceiverConfig {
    pub expected_batches: usize,
    pub maximum_batches: usize,
    pub maximum_message_bytes: usize,
    pub accept_timeout: Duration,
    pub session_timeout: Duration,
    pub read_timeout: Duration,
    pub maximum_detail_records: usize,
}

impl ReceiverConfig {
    /// Validate all caller-controlled limits before allocating or binding.
    pub fn validate(&self) -> Result<()> {
        if self.expected_batches == 0 {
            bail!("expected batch count must be positive");
        }
        if self.maximum_batches == 0 || self.expected_batches > self.maximum_batches {
            bail!("expected batch count exceeds the configured maximum");
        }
        if self.maximum_message_bytes == 0 || self.maximum_message_bytes > u32::MAX as usize {
            bail!("maximum message size must fit a non-zero unsigned 32-bit length");
        }
        if self.accept_timeout.is_zero()
            || self.session_timeout.is_zero()
            || self.read_timeout.is_zero()
        {
            bail!("receiver timeouts must be positive");
        }
        Ok(())
    }
}

/// One bounded per-batch receipt detail record.
#[derive(Debug, Serialize)]
pub struct ReceiptRecord {
    pub batch_sequence: u64,
    pub frame_id: String,
    pub engine_instance_id: String,
    pub source_session_id: String,
    pub outcome: crate::detection_batch::BatchOutcome,
    pub detection_count: usize,
    pub receipt_monotonic_ns: String,
}

#[derive(Debug, Serialize)]
pub struct InterarrivalSummary {
    pub sample_count: usize,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub mean: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProducerIdentity {
    pub engine_instance_id: String,
    pub source_session_id: String,
    pub model_artifact_sha256: String,
    pub source_checkpoint_sha256: Option<String>,
}

impl From<&DetectionBatchV1> for ProducerIdentity {
    fn from(batch: &DetectionBatchV1) -> Self {
        Self {
            engine_instance_id: batch.engine_instance_id.clone(),
            source_session_id: batch.source.session_id.clone(),
            model_artifact_sha256: batch.model.artifact_sha256.clone(),
            source_checkpoint_sha256: batch.model.source_checkpoint_sha256.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ReceiverSettings {
    pub expected_batches: usize,
    pub maximum_batches: usize,
    pub maximum_message_bytes: usize,
    pub accept_timeout_ms: u128,
    pub session_timeout_ms: u128,
    pub read_timeout_ms: u128,
    pub maximum_detail_records: usize,
}

#[derive(Debug, Serialize)]
pub struct RuntimeInfo {
    pub language: &'static str,
    pub application_version: &'static str,
    pub operating_system: &'static str,
    pub architecture: &'static str,
}

/// Versioned machine-readable receipt report. Monotonic values are receiver-
/// local only; the report never subtracts producer clock values from them.
#[derive(Debug, Serialize)]
pub struct ReceiverReport {
    pub schema_version: u8,
    pub receiver_run_id: String,
    pub receiver_process_id: u32,
    pub receiver_clock: &'static str,
    pub cross_process_one_way_latency: &'static str,
    pub expected_batches: usize,
    pub received_batches: usize,
    pub complete: bool,
    pub ordered: bool,
    pub duplicate_sequences: usize,
    pub missing_sequences: usize,
    pub sequence_gap_events: usize,
    pub sequence_gap_total: u64,
    pub out_of_order_records: usize,
    pub unexpected_sequences: usize,
    pub sequence_values: Option<Vec<u64>>,
    pub first_receipt_monotonic_ns: Option<String>,
    pub last_receipt_monotonic_ns: Option<String>,
    pub receipt_window_ms: Option<f64>,
    pub interarrival_ms: InterarrivalSummary,
    pub receipt_record_limit: usize,
    pub receipt_records_retained: usize,
    pub receipt_records_dropped: usize,
    pub receipt_records: Vec<ReceiptRecord>,
    pub errors: Vec<String>,
    pub unexpected_data_after_expected_count: bool,
    pub receiver_elapsed_ms: f64,
    pub producer_identity: Option<ProducerIdentity>,
    pub peer_address: Option<(String, u16)>,
    pub bind_address: String,
    pub batches_jsonl_path: Option<String>,
    pub receiver_settings: ReceiverSettings,
    pub runtime: RuntimeInfo,
    pub batches_jsonl_sha256: Option<String>,
}

impl ReceiverReport {
    /// Add the completed JSONL artifact path and its streaming SHA-256 digest.
    pub async fn attach_jsonl_artifact(&mut self, path: &std::path::Path) -> Result<()> {
        self.batches_jsonl_path = Some(path.to_string_lossy().into_owned());
        self.batches_jsonl_sha256 = Some(sha256_file(path).await?);
        Ok(())
    }

    /// The receiver's successful exit contract requires both completeness and
    /// exact expected sequence order.
    pub fn is_successful(&self) -> bool {
        self.complete && self.ordered
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactRead {
    Complete,
    CleanEof,
    Truncated,
    Timeout,
}

/// Listen on one loopback address and receive exactly one producer session.
///
/// The callback runs after bind so callers can announce an ephemeral port to
/// their producer. Only one connection is accepted; dropping the listener
/// after that accept prevents a second producer from joining the run.
pub async fn serve_once<W, F>(
    bind_address: SocketAddr,
    config: ReceiverConfig,
    jsonl_writer: &mut W,
    ready_callback: F,
) -> Result<(ReceiverReport, SocketAddr)>
where
    W: AsyncWrite + Unpin,
    F: FnOnce(SocketAddr),
{
    config.validate()?;
    ensure_loopback(bind_address)?;

    let listener = TcpListener::bind(bind_address)
        .await
        .with_context(|| format!("failed to bind detection-batch receiver at {bind_address}"))?;
    let bound_address = listener
        .local_addr()
        .context("failed to inspect the bound receiver address")?;
    ready_callback(bound_address);

    let (stream, peer_address) = timeout(config.accept_timeout, listener.accept())
        .await
        .map_err(|_| anyhow!("timed out waiting for a detection-batch producer"))?
        .context("failed to accept detection-batch producer")?;
    drop(listener);

    let mut stream = stream;
    let session = timeout(
        config.session_timeout,
        receive_stream(&mut stream, jsonl_writer, &config),
    )
    .await
    .map_err(|_| anyhow!("detection-batch producer session timed out"))??;

    let mut report = session;
    report.peer_address = Some(peer_address_parts(peer_address));
    report.bind_address = bound_address.to_string();
    Ok((report, bound_address))
}

/// Receive and validate one framed stream without binding a socket.
///
/// This is public so contract tests and other in-process tooling can exercise
/// the exact production parser using a deterministic duplex stream.
pub async fn receive_stream<R, W>(
    reader: &mut R,
    jsonl_writer: &mut W,
    config: &ReceiverConfig,
) -> Result<ReceiverReport>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let started = Instant::now();
    monotonic_now_ns()?;
    let mut receipt_records = Vec::new();
    receipt_records
        .try_reserve(config.maximum_detail_records.min(config.expected_batches))
        .context("could not allocate bounded receipt detail storage")?;

    let mut interarrivals_ms = Vec::new();
    interarrivals_ms
        .try_reserve(config.expected_batches.saturating_sub(1))
        .context("could not allocate bounded interarrival storage")?;

    let mut sequences = HashSet::new();
    sequences
        .try_reserve(config.expected_batches)
        .context("could not allocate bounded sequence tracking storage")?;

    // Keep the complete ordered list only while it fits the caller's detail
    // budget. Aggregate sequence metrics continue after the list is dropped.
    let mut sequence_values = Some(Vec::new());
    let mut stable_identity: Option<ProducerIdentity> = None;
    let mut errors = Vec::new();
    let mut previous_receipt_ns: Option<u64> = None;
    let mut first_receipt_ns: Option<u64> = None;
    let mut last_receipt_ns: Option<u64> = None;
    let mut received_batches = 0usize;
    let mut highest_sequence = 0u64;
    let mut sequence_gap_events = 0usize;
    let mut sequence_gap_total = 0u64;
    let mut out_of_order_records = 0usize;
    let mut unexpected_sequences = 0usize;
    let mut ordered = true;
    let mut unexpected_data = false;

    for batch_index in 0..config.expected_batches {
        let mut header = [0u8; 4];
        match read_exact_with_timeout(reader, &mut header, config.read_timeout).await? {
            ExactRead::Complete => {}
            ExactRead::Timeout => {
                errors.push("read_timeout".to_owned());
                break;
            }
            ExactRead::CleanEof => {
                errors.push("sender_closed_before_expected_batch_count".to_owned());
                break;
            }
            ExactRead::Truncated => {
                errors.push("truncated_frame".to_owned());
                break;
            }
        }

        let payload_size = u32::from_be_bytes(header) as usize;
        if payload_size == 0 {
            errors.push("zero_length_frame".to_owned());
            break;
        }
        if payload_size > config.maximum_message_bytes {
            errors.push("frame_exceeds_configured_maximum".to_owned());
            break;
        }

        let mut payload = Vec::new();
        payload
            .try_reserve_exact(payload_size)
            .context("could not allocate bounded detection-batch payload")?;
        payload.resize(payload_size, 0);
        match read_exact_with_timeout(reader, &mut payload, config.read_timeout).await? {
            ExactRead::Complete => {}
            ExactRead::Timeout => {
                errors.push("read_timeout".to_owned());
                break;
            }
            ExactRead::CleanEof | ExactRead::Truncated => {
                errors.push("truncated_frame".to_owned());
                break;
            }
        }

        let receipt_ns = monotonic_now_ns()?;
        let batch = match decode_detection_batch_v1(&payload) {
            Ok(batch) => batch,
            Err(error) => {
                errors.push(format!("invalid_batch:{error}"));
                break;
            }
        };
        let identity = ProducerIdentity::from(&batch);
        if stable_identity
            .as_ref()
            .is_some_and(|previous| previous != &identity)
        {
            errors.push(
                "invalid_batch:producer, source session, or model identity changed in stream"
                    .to_owned(),
            );
            break;
        }
        stable_identity.get_or_insert_with(|| identity.clone());

        let sequence = batch.batch_sequence;
        if sequence < 1 || sequence > config.expected_batches as u64 {
            unexpected_sequences += 1;
        }
        if !sequences.insert(sequence) {
            // Duplicates are counted after the loop from total valid records
            // minus the unique set size.
        }

        let expected_sequence = u64::try_from(batch_index + 1)
            .context("expected batch sequence does not fit an unsigned 64-bit integer")?;
        ordered &= sequence == expected_sequence;
        if sequence <= highest_sequence {
            out_of_order_records += 1;
        } else if sequence > highest_sequence.saturating_add(1) {
            sequence_gap_events += 1;
            sequence_gap_total = sequence_gap_total
                .saturating_add(sequence.saturating_sub(highest_sequence).saturating_sub(1));
            highest_sequence = sequence;
        } else {
            highest_sequence = sequence;
        }

        if let Some(values) = sequence_values.as_mut() {
            if values.len() < config.maximum_detail_records {
                values.push(sequence);
            } else {
                sequence_values = None;
            }
        }

        if let Some(previous_ns) = previous_receipt_ns {
            interarrivals_ms
                .push(receipt_ns.saturating_sub(previous_ns) as f64 / NANOS_PER_MILLISECOND);
        }
        first_receipt_ns.get_or_insert(receipt_ns);
        last_receipt_ns = Some(receipt_ns);
        previous_receipt_ns = Some(receipt_ns);

        jsonl_writer
            .write_all(&payload)
            .await
            .context("failed to write a received batch to JSONL")?;
        jsonl_writer
            .write_all(b"\n")
            .await
            .context("failed to finish a received JSONL batch")?;
        // Flush each valid record so a bounded valid prefix survives a later
        // sender disconnect or read timeout.
        jsonl_writer
            .flush()
            .await
            .context("failed to flush a received JSONL batch")?;

        received_batches += 1;
        if receipt_records.len() < config.maximum_detail_records {
            receipt_records.push(ReceiptRecord {
                batch_sequence: sequence,
                frame_id: batch.source.frame_id,
                engine_instance_id: batch.engine_instance_id,
                source_session_id: batch.source.session_id,
                outcome: batch.outcome,
                detection_count: batch.detections.len(),
                receipt_monotonic_ns: receipt_ns.to_string(),
            });
        }
    }

    jsonl_writer
        .flush()
        .await
        .context("failed to flush the received JSONL stream")?;

    if received_batches == config.expected_batches && errors.is_empty() {
        let mut extra_byte = [0u8; 1];
        match timeout(config.read_timeout, reader.read(&mut extra_byte)).await {
            Err(_) => errors.push("sender_close_timeout".to_owned()),
            Ok(Err(error)) => return Err(error).context("failed waiting for sender close"),
            Ok(Ok(0)) => {}
            Ok(Ok(_)) => {
                unexpected_data = true;
                errors.push("unexpected_data_after_expected_batch_count".to_owned());
            }
        }
    }

    let unique_expected_sequences = sequences
        .iter()
        .filter(|sequence| **sequence >= 1 && **sequence <= config.expected_batches as u64)
        .count();
    let missing_sequences = config
        .expected_batches
        .saturating_sub(unique_expected_sequences);
    let duplicates = received_batches.saturating_sub(sequences.len());
    ordered &= received_batches == config.expected_batches;
    if received_batches <= config.maximum_detail_records {
        if let Some(values) = sequence_values.as_mut() {
            // Keep an empty list when the caller explicitly set the detail
            // limit to zero and no sequence was received.
            if values.len() != received_batches {
                sequence_values = None;
            }
        }
    } else {
        sequence_values = None;
    }

    let interarrival_ms = summarize_interarrivals(interarrivals_ms);
    let receipt_window_ms = match (first_receipt_ns, last_receipt_ns) {
        (Some(first), Some(last)) => {
            Some(last.saturating_sub(first) as f64 / NANOS_PER_MILLISECOND)
        }
        _ => None,
    };
    let complete = received_batches == config.expected_batches && errors.is_empty();

    Ok(ReceiverReport {
        schema_version: RECEIVER_REPORT_SCHEMA_VERSION,
        receiver_run_id: Uuid::new_v4().to_string(),
        receiver_process_id: std::process::id(),
        receiver_clock: "rust clock_gettime(CLOCK_MONOTONIC); local receipt times only",
        cross_process_one_way_latency: "unavailable_until_clock_epochs_are_verified",
        expected_batches: config.expected_batches,
        received_batches,
        complete,
        ordered,
        duplicate_sequences: duplicates,
        missing_sequences,
        sequence_gap_events,
        sequence_gap_total,
        out_of_order_records,
        unexpected_sequences,
        sequence_values,
        first_receipt_monotonic_ns: first_receipt_ns.map(|value| value.to_string()),
        last_receipt_monotonic_ns: last_receipt_ns.map(|value| value.to_string()),
        receipt_window_ms,
        interarrival_ms,
        receipt_record_limit: config.maximum_detail_records,
        receipt_records_retained: receipt_records.len(),
        receipt_records_dropped: received_batches.saturating_sub(receipt_records.len()),
        receipt_records,
        errors,
        unexpected_data_after_expected_count: unexpected_data,
        receiver_elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
        producer_identity: stable_identity,
        peer_address: None,
        bind_address: String::new(),
        batches_jsonl_path: None,
        receiver_settings: ReceiverSettings {
            expected_batches: config.expected_batches,
            maximum_batches: config.maximum_batches,
            maximum_message_bytes: config.maximum_message_bytes,
            accept_timeout_ms: config.accept_timeout.as_millis(),
            session_timeout_ms: config.session_timeout.as_millis(),
            read_timeout_ms: config.read_timeout.as_millis(),
            maximum_detail_records: config.maximum_detail_records,
        },
        runtime: RuntimeInfo {
            language: "rust",
            application_version: env!("CARGO_PKG_VERSION"),
            operating_system: std::env::consts::OS,
            architecture: std::env::consts::ARCH,
        },
        batches_jsonl_sha256: None,
    })
}

async fn read_exact_with_timeout<R>(
    reader: &mut R,
    buffer: &mut [u8],
    read_timeout: Duration,
) -> Result<ExactRead>
where
    R: AsyncRead + Unpin,
{
    match timeout(read_timeout, read_exact_until_eof(reader, buffer)).await {
        Err(_) => Ok(ExactRead::Timeout),
        Ok(result) => result,
    }
}

async fn read_exact_until_eof<R>(reader: &mut R, buffer: &mut [u8]) -> Result<ExactRead>
where
    R: AsyncRead + Unpin,
{
    let mut read = 0;
    while read < buffer.len() {
        let count = reader
            .read(&mut buffer[read..])
            .await
            .context("failed to read detection-batch frame")?;
        if count == 0 {
            return Ok(if read == 0 {
                ExactRead::CleanEof
            } else {
                ExactRead::Truncated
            });
        }
        read += count;
    }
    Ok(ExactRead::Complete)
}

fn ensure_loopback(address: SocketAddr) -> Result<()> {
    if !address.ip().is_loopback() {
        bail!("receiver address must be loopback");
    }
    Ok(())
}

fn summarize_interarrivals(mut values: Vec<f64>) -> InterarrivalSummary {
    values.sort_by(f64::total_cmp);
    let percentile = |rank: f64| {
        if values.is_empty() {
            None
        } else {
            let index = ((rank * values.len() as f64).ceil() as usize).saturating_sub(1);
            values.get(index.min(values.len() - 1)).copied()
        }
    };
    InterarrivalSummary {
        sample_count: values.len(),
        p50: percentile(0.50),
        p95: percentile(0.95),
        p99: percentile(0.99),
        mean: (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64),
    }
}

fn monotonic_now_ns() -> Result<u64> {
    // Rust's `Instant` intentionally hides its platform epoch. This receiver
    // needs local monotonic receipt values, so use the POSIX monotonic clock on
    // the project's supported macOS and Linux hosts. Cross-process comparisons
    // remain explicitly unavailable until epochs are independently qualified.
    let mut value = std::mem::MaybeUninit::<libc::timespec>::uninit();
    // SAFETY: `clock_gettime` initializes the supplied timespec on success.
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, value.as_mut_ptr()) };
    if result != 0 {
        return Err(io::Error::last_os_error()).context("could not read monotonic clock");
    }
    // SAFETY: successful `clock_gettime` initialized the full timespec.
    let value = unsafe { value.assume_init() };
    let seconds = u64::try_from(value.tv_sec).context("monotonic clock seconds are negative")?;
    let nanoseconds =
        u64::try_from(value.tv_nsec).context("monotonic clock nanoseconds are negative")?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|base| base.checked_add(nanoseconds))
        .ok_or_else(|| anyhow!("monotonic clock value exceeds unsigned 64-bit nanoseconds"))
}

/// Hash a completed JSONL output without loading the whole artifact in memory.
pub async fn sha256_file(path: &std::path::Path) -> Result<String> {
    let mut file = tokio::fs::File::open(path)
        .await
        .with_context(|| format!("failed to reopen received batches at {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut block = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut block)
            .await
            .with_context(|| format!("failed to hash received batches at {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    let encoded = format!("{:x}", digest.finalize());
    debug_assert_eq!(encoded.len(), SHA256_HEX_CHARACTERS);
    Ok(encoded)
}

/// Return the peer address in the historical JSON array representation.
pub fn peer_address_parts(address: SocketAddr) -> (String, u16) {
    (address.ip().to_string(), address.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receiver_limits_reject_unbounded_or_invalid_values() {
        let mut config = ReceiverConfig {
            expected_batches: 1,
            maximum_batches: 10,
            maximum_message_bytes: 1024,
            accept_timeout: Duration::from_secs(1),
            session_timeout: Duration::from_secs(2),
            read_timeout: Duration::from_secs(1),
            maximum_detail_records: 10,
        };
        assert!(config.validate().is_ok());
        config.expected_batches = 11;
        assert!(config.validate().is_err());
        config.expected_batches = 1;
        config.maximum_message_bytes = 0;
        assert!(config.validate().is_err());
        assert!(ensure_loopback("192.0.2.1:9000".parse().unwrap()).is_err());
        assert!(ensure_loopback("[::1]:0".parse().unwrap()).is_ok());
    }

    #[test]
    fn percentile_summary_uses_nearest_rank_and_reports_empty_samples() {
        let summary = summarize_interarrivals(vec![10.0, 30.0, 20.0]);
        assert_eq!(summary.sample_count, 3);
        assert_eq!(summary.p50, Some(20.0));
        assert_eq!(summary.p95, Some(30.0));
        assert_eq!(summary.p99, Some(30.0));
        assert_eq!(summary.mean, Some(20.0));

        let empty = summarize_interarrivals(Vec::new());
        assert_eq!(empty.p50, None);
        assert_eq!(empty.mean, None);
    }
}
