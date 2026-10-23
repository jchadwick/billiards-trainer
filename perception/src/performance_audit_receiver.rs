//! Bounded WebSocket receiver for auditing Rust live-state delivery.
//!
//! This first native collector slice measures receiver-side ordering,
//! observation identity, freshness evidence, and producer-reported
//! read-completion-to-publication age. It deliberately does not subtract a
//! producer monotonic timestamp from a receiver clock: the current live
//! envelope does not establish that those clocks share an origin.

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::Value;
use tokio::time::{interval_at, sleep_until, timeout, Instant as TokioInstant, MissedTickBehavior};
use tokio_tungstenite::tungstenite::{protocol::WebSocketConfig, Message};
use tokio_tungstenite::{connect_async_with_config, WebSocketStream};

use crate::process_resource_sampler::{
    ProcessResourceSampler, ProcessResourceSamplerConfig, ProcessResourceSummary,
};

pub const AUDIT_REPORT_SCHEMA_VERSION: u8 = 3;
pub const DEFAULT_MAXIMUM_MESSAGE_BYTES: usize = 1_048_576;
pub const MAXIMUM_MESSAGE_BYTES: usize = 32 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_SAMPLES: usize = 10_000;
pub const MAXIMUM_SAMPLES: usize = 1_000_000;

#[derive(Debug, Clone)]
pub struct StateReceiverConfig {
    pub duration: Duration,
    pub connect_timeout: Duration,
    pub summary_interval: Duration,
    pub maximum_message_bytes: usize,
    pub maximum_samples: usize,
    /// Optional PID to sample on a separate worker thread. No sampler is
    /// created unless this is explicitly configured.
    pub resource_sample_process_id: Option<u32>,
    pub resource_sample_interval: Duration,
}

impl Default for StateReceiverConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            summary_interval: Duration::from_secs(5),
            maximum_message_bytes: DEFAULT_MAXIMUM_MESSAGE_BYTES,
            maximum_samples: DEFAULT_MAXIMUM_SAMPLES,
            resource_sample_process_id: None,
            resource_sample_interval: Duration::from_secs(1),
        }
    }
}

impl StateReceiverConfig {
    pub fn validate(&self) -> Result<()> {
        if self.duration.is_zero() {
            bail!("audit duration must be positive");
        }
        if self.connect_timeout.is_zero() {
            bail!("WebSocket connect timeout must be positive");
        }
        if self.summary_interval.is_zero() {
            bail!("terminal summary interval must be positive");
        }
        if !(1..=MAXIMUM_MESSAGE_BYTES).contains(&self.maximum_message_bytes) {
            bail!("maximum message bytes must be between 1 and {MAXIMUM_MESSAGE_BYTES}");
        }
        if !(1..=MAXIMUM_SAMPLES).contains(&self.maximum_samples) {
            bail!("maximum samples must be between 1 and {MAXIMUM_SAMPLES}");
        }
        if self.resource_sample_process_id == Some(0) {
            bail!("resource sample process ID must be positive");
        }
        if self.resource_sample_process_id.is_some() && self.resource_sample_interval.is_zero() {
            bail!("resource sample interval must be positive when process sampling is enabled");
        }
        if self.resource_sample_process_id.is_some()
            && self.resource_sample_interval < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL
        {
            bail!(
                "resource sample interval must be at least {} ms for valid process CPU samples",
                sysinfo::MINIMUM_CPU_UPDATE_INTERVAL.as_millis()
            );
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct StateReceiverOutput {
    pub report: StateReceiverReport,
    /// A sliding, bounded detail window. The total message counters in the
    /// report continue to increase after this list reaches its configured cap.
    pub retained_samples: Vec<StateSample>,
}

#[derive(Debug, Serialize)]
pub struct StateReceiverReport {
    pub schema_version: u8,
    pub tool: &'static str,
    pub mode: &'static str,
    pub started_at_utc: String,
    pub finished_at_utc: String,
    pub elapsed_seconds: f64,
    pub termination_reason: &'static str,
    pub counts: MessageCounts,
    pub freshness: FreshnessCounts,
    pub stream_order: SequenceSummary,
    pub source_observations: SourceObservationSummary,
    pub timing: TimingSummary,
    pub cross_process_source_to_receipt_latency: UnavailableMeasurement,
    pub detail_retention: DetailRetention,
    /// Present only when process sampling was explicitly enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_resources: Option<ProcessResourceSummary>,
}

#[derive(Debug, Serialize)]
pub struct MessageCounts {
    pub text_messages: u64,
    pub state_messages: u64,
    pub non_state_messages: u64,
    pub invalid_json_messages: u64,
    pub binary_messages: u64,
}

#[derive(Debug, Serialize)]
pub struct FreshnessCounts {
    pub fresh_state_messages: u64,
    pub stale_state_messages: u64,
    pub unknown_freshness_state_messages: u64,
    pub fresh_unique_source_observations: u64,
}

#[derive(Debug, Serialize)]
pub struct SequenceSummary {
    pub messages_with_sequence: u64,
    pub sequence_gaps: u64,
    pub missing_sequence_total: u64,
    pub duplicate_high_water_sequences: u64,
    pub out_of_order_sequences: u64,
    pub highest_sequence: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct SourceObservationSummary {
    pub messages_with_identity: u64,
    pub unique_source_observations_within_window: u64,
    pub repeated_source_observations_within_window: u64,
    pub messages_without_source_identity: u64,
    pub identity_window_capacity: usize,
    pub identity_window_evictions: u64,
    pub source_frame_sequence_gaps: u64,
    pub missing_source_frame_sequence_total: u64,
}

#[derive(Debug, Serialize)]
pub struct TimingSummary {
    /// Uses the producer-reported frame read completion to public-state
    /// publication age. It is not camera-exposure-to-publication latency.
    pub read_completion_to_publication_age_ms: PercentileSummary,
    /// Calculated entirely from this receiver process's local monotonic clock.
    pub local_state_receive_interval_ms: PercentileSummary,
}

#[derive(Debug, Serialize)]
pub struct PercentileSummary {
    pub sample_count: u64,
    pub retained_sample_count: usize,
    pub sample_limit: usize,
    pub truncated: bool,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct UnavailableMeasurement {
    pub status: &'static str,
    pub reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct DetailRetention {
    pub sample_limit: usize,
    pub retained_state_samples: usize,
    pub state_samples_truncated: bool,
}

#[derive(Debug, Serialize, Clone)]
pub struct StateSample {
    pub local_receive_elapsed_ms: f64,
    pub stream_sequence: Option<u64>,
    pub transport_sequence: Option<u64>,
    pub observation_identity: Option<String>,
    pub source_frame_id: Option<String>,
    pub source_timestamp_kind: Option<String>,
    pub freshness_status: Option<String>,
    pub ball_positions_status: Option<String>,
    pub ball_count: Option<u64>,
    pub configuration_revision: Option<u64>,
    pub rectification_revision: Option<u64>,
    pub calibration_revision: Option<u64>,
    pub read_completion_to_publication_age_ms: Option<f64>,
}

/// Connect to one anonymous, plain-WebSocket endpoint and receive for a
/// bounded duration. Userinfo is rejected and no auth headers or credentials
/// are supported by this tool.
pub async fn connect_and_collect(
    url: &str,
    config: &StateReceiverConfig,
) -> Result<StateReceiverOutput> {
    config.validate()?;
    validate_websocket_url(url)?;

    let websocket_config = WebSocketConfig::default()
        .max_message_size(Some(config.maximum_message_bytes))
        .max_frame_size(Some(config.maximum_message_bytes));
    let (socket, _response) = timeout(
        config.connect_timeout,
        connect_async_with_config(url, Some(websocket_config), false),
    )
    .await
    .context("WebSocket connection timed out")?
    .context("could not connect to the WebSocket state endpoint")?;
    collect_connected_stream(socket, config).await
}

/// Collect from an already-connected stream. Public for contract tests that
/// use an in-process WebSocket peer instead of the application gateway.
pub async fn collect_connected_stream<S>(
    mut socket: WebSocketStream<S>,
    config: &StateReceiverConfig,
) -> Result<StateReceiverOutput>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    config.validate()?;
    let started_at_utc = utc_now();
    let started = Instant::now();
    let mut resource_sampler = config
        .resource_sample_process_id
        .map(|process_id| {
            ProcessResourceSampler::start(ProcessResourceSamplerConfig {
                process_id,
                sample_interval: config.resource_sample_interval,
                maximum_samples: config.maximum_samples,
            })
        })
        .transpose()?;
    let deadline = TokioInstant::now() + config.duration;
    let mut summary_tick = interval_at(
        TokioInstant::now() + config.summary_interval,
        config.summary_interval,
    );
    summary_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut accumulator = StateAccumulator::new(config.maximum_samples);
    let termination_reason = loop {
        tokio::select! {
            _ = sleep_until(deadline) => break "duration_elapsed",
            _ = summary_tick.tick() => {
                eprintln!(
                    "audit_live_summary states={} fresh={} unique_sources={} repeated_sources={} invalid_json={}",
                    accumulator.counts.state_messages,
                    accumulator.freshness.fresh_state_messages,
                    accumulator.source.unique,
                    accumulator.source.repeated,
                    accumulator.counts.invalid_json_messages,
                );
            }
            message = socket.next() => match message {
                Some(Ok(Message::Text(text))) => accumulator.record_payload(text.as_bytes(), started.elapsed(), true),
                Some(Ok(Message::Binary(bytes))) => {
                    accumulator.counts.binary_messages = accumulator.counts.binary_messages.saturating_add(1);
                    if bytes.len() <= config.maximum_message_bytes {
                        accumulator.record_payload(&bytes, started.elapsed(), false);
                    }
                }
                Some(Ok(Message::Close(_))) | None => break "peer_closed",
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) | Some(Ok(Message::Frame(_))) => {}
                Some(Err(error)) => return Err(error).context("WebSocket state receive failed"),
            }
        }
    };

    let elapsed = started.elapsed();
    let _ = socket.close(None).await;
    let finished_at_utc = utc_now();
    let process_resources = resource_sampler
        .as_mut()
        .map(|sampler| {
            sampler.stop()?;
            Ok::<ProcessResourceSummary, anyhow::Error>(sampler.summary())
        })
        .transpose()?;
    Ok(accumulator.finish(
        started_at_utc,
        finished_at_utc,
        elapsed,
        termination_reason,
        config.maximum_samples,
        process_resources,
    ))
}

pub fn validate_websocket_url(url: &str) -> Result<()> {
    let Some(authority_and_path) = url.strip_prefix("ws://") else {
        bail!("WebSocket URL must use ws://; secure or authenticated endpoints are not supported by this local audit tool");
    };
    let authority = authority_and_path.split('/').next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') || url.contains('#') || url.contains('?') {
        bail!("WebSocket URL must have a host and contain no userinfo, query, or fragment");
    }
    if url.len() > 4096 {
        bail!("WebSocket URL exceeds the maximum length of 4096 bytes");
    }
    Ok(())
}

fn utc_now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

#[derive(Debug, Default)]
struct MutableCounts {
    text_messages: u64,
    state_messages: u64,
    non_state_messages: u64,
    invalid_json_messages: u64,
    binary_messages: u64,
}

#[derive(Debug, Default)]
struct MutableFreshness {
    fresh_state_messages: u64,
    stale_state_messages: u64,
    unknown_freshness_state_messages: u64,
    fresh_unique_source_observations: u64,
}

#[derive(Debug, Default)]
struct SequenceAccumulator {
    messages_with_sequence: u64,
    gaps: u64,
    missing_total: u64,
    duplicates: u64,
    out_of_order: u64,
    high_water: Option<u64>,
}

impl SequenceAccumulator {
    fn record(&mut self, sequence: Option<u64>) {
        let Some(sequence) = sequence else {
            return;
        };
        self.messages_with_sequence = self.messages_with_sequence.saturating_add(1);
        match self.high_water {
            Some(high_water) if sequence == high_water => {
                self.duplicates = self.duplicates.saturating_add(1);
            }
            Some(high_water) if sequence < high_water => {
                self.out_of_order = self.out_of_order.saturating_add(1);
            }
            Some(high_water) => {
                let missing = sequence.saturating_sub(high_water).saturating_sub(1);
                if missing > 0 {
                    self.gaps = self.gaps.saturating_add(1);
                    self.missing_total = self.missing_total.saturating_add(missing);
                }
                self.high_water = Some(sequence);
            }
            None => self.high_water = Some(sequence),
        }
    }
}

#[derive(Debug, Default)]
struct SourceAccumulator {
    messages_with_identity: u64,
    unique: u64,
    repeated: u64,
    unavailable: u64,
    evictions: u64,
    frame_gaps: u64,
    missing_frames: u64,
    last_numeric_frame: Option<u64>,
    last_numeric_scope: Option<String>,
    identities: HashSet<String>,
    identity_order: VecDeque<String>,
    fresh_identities: HashSet<String>,
}

impl SourceAccumulator {
    fn record(
        &mut self,
        identity: Option<&str>,
        numeric_frame: Option<u64>,
        scope: Option<&str>,
        is_fresh: bool,
        capacity: usize,
    ) -> bool {
        let Some(identity) = identity else {
            self.unavailable = self.unavailable.saturating_add(1);
            return false;
        };
        self.messages_with_identity = self.messages_with_identity.saturating_add(1);
        let is_new_identity = !self.identities.contains(identity);
        if !is_new_identity {
            self.repeated = self.repeated.saturating_add(1);
        } else {
            if self.identity_order.len() == capacity {
                if let Some(evicted) = self.identity_order.pop_front() {
                    self.identities.remove(&evicted);
                    self.fresh_identities.remove(&evicted);
                    self.evictions = self.evictions.saturating_add(1);
                }
            }
            self.identities.insert(identity.to_owned());
            self.identity_order.push_back(identity.to_owned());
            self.unique = self.unique.saturating_add(1);
        }
        let fresh_unique = is_fresh && self.fresh_identities.insert(identity.to_owned());
        if is_new_identity {
            if let Some(frame) = numeric_frame {
                // Frame numbers commonly restart when the camera/session restarts.
                // Never report a gap across unrelated source identity scopes.
                if self.last_numeric_scope.as_deref() != scope {
                    self.last_numeric_frame = None;
                    self.last_numeric_scope = scope.map(str::to_owned);
                }
                if let Some(previous) = self.last_numeric_frame {
                    let missing = frame.saturating_sub(previous).saturating_sub(1);
                    if frame > previous && missing > 0 {
                        self.frame_gaps = self.frame_gaps.saturating_add(1);
                        self.missing_frames = self.missing_frames.saturating_add(missing);
                    }
                }
                self.last_numeric_frame =
                    Some(self.last_numeric_frame.map_or(frame, |old| old.max(frame)));
            }
        }
        fresh_unique
    }
}

#[derive(Debug)]
struct StateAccumulator {
    counts: MutableCounts,
    freshness: MutableFreshness,
    sequence: SequenceAccumulator,
    source: SourceAccumulator,
    last_state_receive: Option<Duration>,
    receive_intervals_ms: VecDeque<f64>,
    receive_interval_count: u64,
    publication_ages_ms: VecDeque<f64>,
    publication_age_count: u64,
    samples: VecDeque<StateSample>,
    maximum_samples: usize,
}

impl StateAccumulator {
    fn new(maximum_samples: usize) -> Self {
        Self {
            counts: MutableCounts::default(),
            freshness: MutableFreshness::default(),
            sequence: SequenceAccumulator::default(),
            source: SourceAccumulator::default(),
            last_state_receive: None,
            receive_intervals_ms: VecDeque::with_capacity(maximum_samples),
            receive_interval_count: 0,
            publication_ages_ms: VecDeque::with_capacity(maximum_samples),
            publication_age_count: 0,
            samples: VecDeque::with_capacity(maximum_samples),
            maximum_samples,
        }
    }

    #[cfg(test)]
    fn record_text(&mut self, bytes: &[u8], received_after: Duration) {
        self.record_payload(bytes, received_after, true);
    }

    fn record_payload(&mut self, bytes: &[u8], received_after: Duration, is_text: bool) {
        if is_text {
            self.counts.text_messages = self.counts.text_messages.saturating_add(1);
        }
        let message = match serde_json::from_slice::<Value>(bytes) {
            Ok(message) => message,
            Err(_) => {
                self.counts.invalid_json_messages =
                    self.counts.invalid_json_messages.saturating_add(1);
                return;
            }
        };
        if message.get("type").and_then(Value::as_str) != Some("state") {
            self.counts.non_state_messages = self.counts.non_state_messages.saturating_add(1);
            return;
        }
        self.counts.state_messages = self.counts.state_messages.saturating_add(1);

        if let Some(previous) = self.last_state_receive {
            record_sample(
                &mut self.receive_intervals_ms,
                &mut self.receive_interval_count,
                self.maximum_samples,
                received_after.saturating_sub(previous).as_secs_f64() * 1_000.0,
            );
        }
        self.last_state_receive = Some(received_after);

        let data = message.get("data").unwrap_or(&Value::Null);
        let status = data
            .get("observation_status")
            .or_else(|| data.get("status"))
            .unwrap_or(&Value::Null);
        let identity_value = status.get("observation_identity").unwrap_or(&Value::Null);
        let engine_id = identity_value
            .get("engine_instance_id")
            .or_else(|| status.get("engine_instance_id"))
            .or_else(|| status.get("vision_instance_id"))
            .and_then(Value::as_str);
        let session_id = identity_value
            .get("session_id")
            .or_else(|| status.get("source_session_id"))
            .and_then(Value::as_str);
        let source_frame_id = identity_value
            .get("frame_id")
            .or_else(|| status.get("source_frame_id"))
            .or_else(|| status.get("source_frame_number"))
            .and_then(string_value);
        let observation_identity = match (engine_id, session_id, source_frame_id.as_deref()) {
            (Some(engine), Some(session), Some(frame)) => {
                Some(format!("{engine}\u{1f}{session}\u{1f}{frame}"))
            }
            _ => None,
        };
        let source_scope = match (engine_id, session_id) {
            (Some(engine), Some(session)) => Some(format!("{engine}\u{1f}{session}")),
            _ => None,
        };
        let numeric_frame = source_frame_id.as_deref().and_then(numeric_frame_sequence);
        let freshness_status = string_at(status, "freshness_status");
        let is_fresh = freshness_status.as_deref() == Some("fresh");
        let fresh_source = self.source.record(
            observation_identity.as_deref(),
            numeric_frame,
            source_scope.as_deref(),
            is_fresh,
            self.maximum_samples,
        );

        match freshness_status.as_deref() {
            Some("fresh") => {
                self.freshness.fresh_state_messages =
                    self.freshness.fresh_state_messages.saturating_add(1);
                if fresh_source {
                    self.freshness.fresh_unique_source_observations = self
                        .freshness
                        .fresh_unique_source_observations
                        .saturating_add(1);
                }
            }
            Some("stale") | Some("expired") => {
                self.freshness.stale_state_messages =
                    self.freshness.stale_state_messages.saturating_add(1);
            }
            _ => {
                self.freshness.unknown_freshness_state_messages = self
                    .freshness
                    .unknown_freshness_state_messages
                    .saturating_add(1);
            }
        }

        let stream_sequence =
            unsigned_at(&message, "sequence").or_else(|| unsigned_at(data, "sequence"));
        self.sequence.record(stream_sequence);
        let transport_sequence =
            unsigned_at(status, "transport_sequence").or_else(|| unsigned_at(data, "frame_number"));
        let age_ms = finite_number_at(status, "frame_publication_age_seconds")
            .filter(|age| *age >= 0.0)
            .map(|age| age * 1_000.0);
        if let Some(age) = age_ms {
            record_sample(
                &mut self.publication_ages_ms,
                &mut self.publication_age_count,
                self.maximum_samples,
                age,
            );
        }
        let balls = data.get("balls").and_then(Value::as_array);
        let ball_count =
            unsigned_at(data, "ball_count").or_else(|| balls.map(|items| items.len() as u64));
        let sample = StateSample {
            local_receive_elapsed_ms: received_after.as_secs_f64() * 1_000.0,
            stream_sequence,
            transport_sequence,
            observation_identity,
            source_frame_id,
            source_timestamp_kind: string_at(status, "source_timestamp_kind")
                .or_else(|| string_at(status, "ball_observation_source_timestamp_kind")),
            freshness_status,
            ball_positions_status: string_at(status, "ball_positions_status")
                .or_else(|| string_at(status, "ball_detection_status")),
            ball_count,
            configuration_revision: unsigned_at(status, "configuration_revision"),
            rectification_revision: unsigned_at(status, "rectification_revision"),
            calibration_revision: unsigned_at(status, "calibration_revision"),
            read_completion_to_publication_age_ms: age_ms,
        };
        self.push_detail(sample);
    }

    fn push_detail(&mut self, sample: StateSample) {
        if self.samples.len() == self.maximum_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    fn finish(
        self,
        started_at_utc: String,
        finished_at_utc: String,
        elapsed: Duration,
        termination_reason: &'static str,
        maximum_samples: usize,
        process_resources: Option<ProcessResourceSummary>,
    ) -> StateReceiverOutput {
        let retained_sample_count = self.samples.len();
        let state_samples_truncated = self.counts.state_messages > retained_sample_count as u64;
        let report = StateReceiverReport {
            schema_version: AUDIT_REPORT_SCHEMA_VERSION,
            tool: "billiards-performance-audit",
            mode: "websocket_state_receiver",
            started_at_utc,
            finished_at_utc,
            elapsed_seconds: elapsed.as_secs_f64(),
            termination_reason,
            counts: MessageCounts {
                text_messages: self.counts.text_messages,
                state_messages: self.counts.state_messages,
                non_state_messages: self.counts.non_state_messages,
                invalid_json_messages: self.counts.invalid_json_messages,
                binary_messages: self.counts.binary_messages,
            },
            freshness: FreshnessCounts {
                fresh_state_messages: self.freshness.fresh_state_messages,
                stale_state_messages: self.freshness.stale_state_messages,
                unknown_freshness_state_messages: self.freshness.unknown_freshness_state_messages,
                fresh_unique_source_observations: self.freshness.fresh_unique_source_observations,
            },
            stream_order: SequenceSummary {
                messages_with_sequence: self.sequence.messages_with_sequence,
                sequence_gaps: self.sequence.gaps,
                missing_sequence_total: self.sequence.missing_total,
                duplicate_high_water_sequences: self.sequence.duplicates,
                out_of_order_sequences: self.sequence.out_of_order,
                highest_sequence: self.sequence.high_water,
            },
            source_observations: SourceObservationSummary {
                messages_with_identity: self.source.messages_with_identity,
                unique_source_observations_within_window: self.source.unique,
                repeated_source_observations_within_window: self.source.repeated,
                messages_without_source_identity: self.source.unavailable,
                identity_window_capacity: maximum_samples,
                identity_window_evictions: self.source.evictions,
                source_frame_sequence_gaps: self.source.frame_gaps,
                missing_source_frame_sequence_total: self.source.missing_frames,
            },
            timing: TimingSummary {
                read_completion_to_publication_age_ms: percentile_summary(
                    &self.publication_ages_ms,
                    self.publication_age_count,
                    maximum_samples,
                ),
                local_state_receive_interval_ms: percentile_summary(
                    &self.receive_intervals_ms,
                    self.receive_interval_count,
                    maximum_samples,
                ),
            },
            cross_process_source_to_receipt_latency: UnavailableMeasurement {
                status: "unavailable_unverified_clock",
                reason:
                    "producer monotonic ticks are not proven comparable with the receiver clock",
            },
            detail_retention: DetailRetention {
                sample_limit: maximum_samples,
                retained_state_samples: retained_sample_count,
                state_samples_truncated,
            },
            process_resources,
        };
        StateReceiverOutput {
            report,
            retained_samples: self.samples.into_iter().collect(),
        }
    }
}

fn percentile_summary(
    values: &VecDeque<f64>,
    total_count: u64,
    sample_limit: usize,
) -> PercentileSummary {
    let mut ordered: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    ordered.sort_by(f64::total_cmp);
    PercentileSummary {
        sample_count: total_count,
        retained_sample_count: ordered.len(),
        sample_limit,
        truncated: total_count > ordered.len() as u64,
        p50: percentile(&ordered, 0.50),
        p95: percentile(&ordered, 0.95),
        p99: percentile(&ordered, 0.99),
    }
}

fn record_sample(values: &mut VecDeque<f64>, total: &mut u64, capacity: usize, value: f64) {
    if !value.is_finite() {
        return;
    }
    *total = total.saturating_add(1);
    if values.len() == capacity {
        values.pop_front();
    }
    values.push_back(value);
}

fn percentile(ordered: &[f64], percentile: f64) -> Option<f64> {
    if ordered.is_empty() {
        return None;
    }
    if ordered.len() == 1 {
        return ordered.first().copied();
    }
    let rank = percentile * (ordered.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    Some(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)
}

fn string_at(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn string_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

/// Rust scene IDs use `frame-N` (zero-padded in replay). Accept the trailing
/// decimal sequence as the source-frame counter while keeping the full ID for
/// deduplication. Bare numeric IDs remain supported for older publishers.
fn numeric_frame_sequence(frame_id: &str) -> Option<u64> {
    let sequence = frame_id.rsplit('-').next()?;
    if sequence.is_empty() || !sequence.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    sequence.parse().ok()
}

fn unsigned_at(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn finite_number_at(value: &Value, key: &str) -> Option<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
}

/// Unix-epoch milliseconds from the receiving host, available for audit event
/// timestamps. This value is not subtracted from the producer monotonic clock.
pub fn receiver_wall_time_ms() -> Option<u128> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|age| age.as_millis())
}

#[cfg(test)]
mod tests {
    use super::{
        numeric_frame_sequence, validate_websocket_url, SequenceAccumulator, SourceAccumulator,
        StateAccumulator, StateReceiverConfig,
    };
    use serde_json::json;
    use std::time::{Duration, Instant};

    fn state_message(sequence: u64, frame: &str, freshness: &str, age_seconds: f64) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "type": "state",
            "sequence": sequence,
            "data": {
                "sequence": sequence,
                "frame_number": sequence,
                "ball_count": 3,
                "balls": [{}, {}, {}],
                "observation_status": {
                    "freshness_status": freshness,
                    "ball_positions_status": "observed",
                    "frame_publication_age_seconds": age_seconds,
                    "source_timestamp_kind": "monotonic:engine-a:camera_read_complete",
                    "configuration_revision": 7,
                    "rectification_revision": 2,
                    "calibration_revision": 9,
                    "observation_identity": {
                        "engine_instance_id": "engine-a",
                        "session_id": "session-a",
                        "frame_id": frame,
                    }
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn config_is_bounded_and_requires_positive_limits() {
        let mut config = StateReceiverConfig::default();
        assert!(config.validate().is_ok());
        config.maximum_samples = 0;
        assert!(config.validate().is_err());
        config.maximum_samples = 1;
        config.maximum_message_bytes = super::MAXIMUM_MESSAGE_BYTES + 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn resource_sampling_is_opt_in_and_requires_valid_pid_and_cpu_interval() {
        let mut config = StateReceiverConfig::default();
        assert!(config.validate().is_ok());
        config.resource_sample_process_id = Some(0);
        assert!(config.validate().is_err());
        config.resource_sample_process_id = Some(1);
        config.resource_sample_interval = Duration::from_millis(1);
        assert!(config.validate().is_err());
        config.resource_sample_interval = sysinfo::MINIMUM_CPU_UPDATE_INTERVAL;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_credentialed_fragmented_and_non_ws_urls() {
        for url in [
            "wss://localhost/state",
            "ws://user:secret@localhost/state",
            "ws://localhost/state#fragment",
            "ws://localhost/state?token=secret",
            "ws:///state",
        ] {
            assert!(validate_websocket_url(url).is_err(), "accepted {url}");
        }
        assert!(validate_websocket_url("ws://127.0.0.1:8001/api/v1/websocket/ws").is_ok());
    }

    #[test]
    fn counts_fresh_and_repeated_source_frames_with_bounded_details() {
        let mut accumulator = StateAccumulator::new(2);
        accumulator.record_text(
            &state_message(1, "10", "fresh", 0.01),
            Duration::from_millis(1),
        );
        accumulator.record_text(
            &state_message(2, "10", "fresh", 0.02),
            Duration::from_millis(2),
        );
        accumulator.record_text(
            &state_message(3, "11", "fresh", 0.03),
            Duration::from_millis(3),
        );
        let output = accumulator.finish(
            "start".to_owned(),
            "finish".to_owned(),
            Duration::from_secs(1),
            "peer_closed",
            2,
            None,
        );

        assert_eq!(output.report.counts.state_messages, 3);
        assert_eq!(output.report.freshness.fresh_state_messages, 3);
        assert_eq!(output.report.freshness.fresh_unique_source_observations, 2);
        assert_eq!(
            output
                .report
                .source_observations
                .repeated_source_observations_within_window,
            1
        );
        assert_eq!(output.retained_samples.len(), 2);
        assert!(output.report.detail_retention.state_samples_truncated);
        assert_eq!(
            output
                .report
                .timing
                .read_completion_to_publication_age_ms
                .sample_count,
            3
        );
        assert!(output
            .report
            .cross_process_source_to_receipt_latency
            .status
            .starts_with("unavailable"));
    }

    #[test]
    fn separates_non_state_and_invalid_json_and_reports_stream_order() {
        let mut accumulator = StateAccumulator::new(10);
        accumulator.record_text(b"not json", Duration::from_millis(1));
        accumulator.record_text(br#"{"type":"heartbeat"}"#, Duration::from_millis(2));
        accumulator.record_text(
            &state_message(1, "1", "stale", 0.01),
            Duration::from_millis(3),
        );
        accumulator.record_text(
            &state_message(4, "4", "unknown", 0.01),
            Duration::from_millis(4),
        );
        accumulator.record_text(
            &state_message(4, "2", "fresh", 0.01),
            Duration::from_millis(5),
        );
        accumulator.record_text(
            &state_message(2, "5", "fresh", 0.01),
            Duration::from_millis(6),
        );
        let output = accumulator.finish(
            "start".to_owned(),
            "finish".to_owned(),
            Duration::from_secs(1),
            "peer_closed",
            10,
            None,
        );

        assert_eq!(output.report.counts.invalid_json_messages, 1);
        assert_eq!(output.report.counts.non_state_messages, 1);
        assert_eq!(output.report.stream_order.sequence_gaps, 1);
        assert_eq!(output.report.stream_order.missing_sequence_total, 2);
        assert_eq!(output.report.stream_order.duplicate_high_water_sequences, 1);
        assert_eq!(output.report.stream_order.out_of_order_sequences, 1);
        assert_eq!(output.report.freshness.stale_state_messages, 1);
        assert_eq!(output.report.freshness.unknown_freshness_state_messages, 1);
    }

    #[test]
    fn bounded_identity_window_discloses_eviction() {
        let mut source = SourceAccumulator::default();
        assert!(source.record(Some("one"), Some(1), Some("camera-a"), true, 2));
        assert!(source.record(Some("two"), Some(2), Some("camera-a"), true, 2));
        assert!(!source.record(Some("one"), Some(1), Some("camera-a"), true, 2));
        assert!(source.record(Some("three"), Some(3), Some("camera-a"), true, 2));
        assert!(source.record(Some("one"), Some(1), Some("camera-a"), true, 2));
        assert_eq!(source.evictions, 2);
        assert_eq!(source.unique, 4);
    }

    #[test]
    fn camera_restart_does_not_create_cross_session_frame_gap() {
        let mut source = SourceAccumulator::default();
        assert!(source.record(Some("session-a/100"), Some(100), Some("session-a"), true, 8));
        assert!(source.record(Some("session-b/1"), Some(1), Some("session-b"), true, 8));
        assert_eq!(source.frame_gaps, 0);
        assert!(source.record(Some("session-b/3"), Some(3), Some("session-b"), true, 8));
        assert_eq!(source.frame_gaps, 1);
        assert_eq!(source.missing_frames, 1);
    }

    #[test]
    fn extracts_numeric_suffix_from_rust_frame_identity_for_gap_accounting() {
        let mut accumulator = StateAccumulator::new(8);
        for (sequence, frame_id) in [
            (1, "frame-000100"),
            (2, "frame-000103"),
            (3, "frame-000104"),
        ] {
            accumulator.record_text(
                &state_message(sequence, frame_id, "fresh", 0.01),
                Duration::from_millis(sequence),
            );
        }
        let output = accumulator.finish(
            "start".to_owned(),
            "finish".to_owned(),
            Duration::from_secs(1),
            "peer_closed",
            8,
            None,
        );
        assert_eq!(
            output.report.source_observations.source_frame_sequence_gaps,
            1
        );
        assert_eq!(
            output
                .report
                .source_observations
                .missing_source_frame_sequence_total,
            2
        );
    }

    #[test]
    fn source_frame_counter_parser_accepts_supported_forms_and_rejects_others() {
        for (frame_id, expected) in [
            ("42", Some(42)),
            ("frame-000042", Some(42)),
            ("camera-frame-42", Some(42)),
            ("frame-", None),
            ("frame-x", None),
            ("frame-42x", None),
            ("frame-18446744073709551616", None),
        ] {
            assert_eq!(numeric_frame_sequence(frame_id), expected, "{frame_id}");
        }
    }

    #[test]
    fn stale_then_fresh_publication_counts_a_fresh_unique_observation_once() {
        let mut accumulator = StateAccumulator::new(4);
        accumulator.record_text(
            &state_message(1, "10", "stale", 0.01),
            Duration::from_millis(1),
        );
        accumulator.record_text(
            &state_message(2, "10", "fresh", 0.01),
            Duration::from_millis(2),
        );
        accumulator.record_text(
            &state_message(3, "10", "fresh", 0.01),
            Duration::from_millis(3),
        );
        let output = accumulator.finish(
            "start".to_owned(),
            "finish".to_owned(),
            Duration::from_secs(1),
            "peer_closed",
            4,
            None,
        );
        assert_eq!(
            output
                .report
                .source_observations
                .unique_source_observations_within_window,
            1
        );
        assert_eq!(
            output
                .report
                .source_observations
                .repeated_source_observations_within_window,
            2
        );
        assert_eq!(output.report.freshness.fresh_unique_source_observations, 1);
    }

    #[test]
    fn local_receive_intervals_use_only_the_receiver_monotonic_clock() {
        let mut accumulator = StateAccumulator::new(10);
        let started = Instant::now();
        accumulator.record_text(&state_message(1, "1", "fresh", 0.01), Duration::ZERO);
        accumulator.record_text(&state_message(2, "2", "fresh", 0.01), started.elapsed());
        let output = accumulator.finish(
            "start".to_owned(),
            "finish".to_owned(),
            Duration::from_secs(1),
            "peer_closed",
            10,
            None,
        );
        assert_eq!(
            output
                .report
                .timing
                .local_state_receive_interval_ms
                .sample_count,
            1
        );
        assert_eq!(
            output.report.cross_process_source_to_receipt_latency.status,
            "unavailable_unverified_clock"
        );
    }

    #[test]
    fn sequence_accumulator_keeps_a_high_water_mark() {
        let mut sequence = SequenceAccumulator::default();
        for value in [10, 12, 11, 12] {
            sequence.record(Some(value));
        }
        assert_eq!(sequence.gaps, 1);
        assert_eq!(sequence.missing_total, 1);
        assert_eq!(sequence.out_of_order, 1);
        assert_eq!(sequence.duplicates, 1);
        assert_eq!(sequence.high_water, Some(12));
    }
}
