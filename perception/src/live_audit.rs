//! Optional bounded timing export for the supervised live perception process.
//!
//! The audit stream is a separate, unauthenticated loopback TCP connection. It
//! never carries image bytes, detections, ball coordinates, or scene payloads.
//! When disabled, callers construct no recorder, channel, thread, or socket.

use std::io::{self, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use crate::capture::CaptureFrameTiming;
use crate::capture_pipeline::CapturePipelineStats;
use crate::detection_batch::BatchTimestamp;
use crate::inference::FrameInferenceTimings;
use crate::live_timing_spans::FrameTimingSnapshot;
use crate::SceneTransportStats;

/// Maximum number of queued audit events. Each event contains bounded identity
/// strings and scalar timings, not frame- or scene-sized payloads.
pub const MAX_LIVE_AUDIT_QUEUE_CAPACITY: usize = 4_096;
/// An audit record must be large enough for a complete bounded timing record,
/// and is capped so a malformed or future record cannot monopolize the writer.
pub const MIN_LIVE_AUDIT_RECORD_BYTES: usize = 1_024;
pub const MAX_LIVE_AUDIT_RECORD_BYTES: usize = 1_048_576;
pub const LIVE_AUDIT_SCHEMA_VERSION: u8 = 5;

/// Explicit settings for the opt-in side channel. There are no implicit
/// address, queue, framing, or deadline defaults.
#[derive(Debug, Clone, Copy)]
pub struct LiveAuditConfig {
    pub address: SocketAddr,
    pub queue_capacity: usize,
    pub maximum_record_bytes: usize,
    pub connect_timeout: Duration,
    pub write_timeout: Duration,
}

impl LiveAuditConfig {
    /// Validate all resource bounds before creating the queue or worker.
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.address.ip().is_loopback(),
            "live audit TCP address must be loopback"
        );
        ensure!(
            (1..=MAX_LIVE_AUDIT_QUEUE_CAPACITY).contains(&self.queue_capacity),
            "live audit queue capacity must be between 1 and {MAX_LIVE_AUDIT_QUEUE_CAPACITY}"
        );
        ensure!(
            (MIN_LIVE_AUDIT_RECORD_BYTES..=MAX_LIVE_AUDIT_RECORD_BYTES)
                .contains(&self.maximum_record_bytes),
            "live audit maximum record size must be between {MIN_LIVE_AUDIT_RECORD_BYTES} and {MAX_LIVE_AUDIT_RECORD_BYTES} bytes"
        );
        ensure!(
            !self.connect_timeout.is_zero()
                && Instant::now().checked_add(self.connect_timeout).is_some(),
            "live audit connect deadline must be positive and representable"
        );
        ensure!(
            !self.write_timeout.is_zero()
                && Instant::now().checked_add(self.write_timeout).is_some(),
            "live audit write deadline must be positive and representable"
        );
        Ok(())
    }
}

/// Data needed to publish one correlated frame timing record.
pub(crate) struct LiveAuditFrame<'a> {
    pub engine_instance_id: &'a str,
    pub source_session_id: &'a str,
    pub frame_id: &'a str,
    pub transport_sequence: u64,
    pub configuration_revision: u64,
    pub tracking_configuration_revision: u64,
    pub rectification_revision: u64,
    pub calibration_revision: Option<u64>,
    pub detection_count: usize,
    pub track_count: usize,
    pub source_timestamp: Option<&'a BatchTimestamp>,
    pub capture_timing: crate::capture::CaptureFrameTiming,
    pub parent_record_read_complete_to_inference_start_ms: f64,
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub decode_and_nms_ms: f64,
    pub engine_total_ms: f64,
    pub calibration_ms: Option<f64>,
    pub tracking_state_ms: Option<f64>,
    pub scene_serialization_ms: Option<f64>,
    pub scene_queue_submit_ms: Option<f64>,
    pub parent_record_read_complete_to_scene_queue_ms: f64,
    pub frame_timing: Option<&'a FrameTimingSnapshot>,
}

/// A scalar-only disposition for an observation that did not produce a frame
/// timing record. `transport_sequence` is absent until a scene owns a sequence.
pub(crate) struct LiveAuditObservationOutcome<'a> {
    pub engine_instance_id: &'a str,
    pub source_session_id: &'a str,
    pub frame_id: &'a str,
    pub transport_sequence: Option<u64>,
    pub configuration_revision: u64,
    pub tracking_configuration_revision: u64,
    pub rectification_revision: u64,
    pub calibration_revision: Option<u64>,
    pub source_timestamp: Option<&'a BatchTimestamp>,
    pub outcome: &'static str,
    pub stage: &'static str,
    pub reason: &'a str,
    pub inference_timings: Option<LiveAuditInferenceTimings>,
}

/// Same-parent-clock timing evidence for an inference result that was rejected
/// by the freshness guard. `source_timestamp` remains separately represented
/// by the source's nullable timestamp and is never synthesized here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LiveAuditInferenceTimings {
    pub capture_timing: CaptureFrameTiming,
    pub parent_record_read_complete_to_inference_start_ms: f64,
    pub inference: FrameInferenceTimings,
    pub parent_record_read_complete_to_inference_complete_ms: f64,
    pub maximum_frame_age_ms: f64,
}

pub(crate) struct LiveAuditSceneWrite<'a> {
    pub engine_instance_id: &'a str,
    pub source_session_id: &'a str,
    pub frame_id: &'a str,
    pub transport_sequence: u64,
    pub write_attempt: u64,
    pub duration_ms: f64,
    pub succeeded: bool,
}

/// Per-client gateway send timing measured only when gateway auditing is enabled.
pub(crate) struct LiveAuditGatewayWebSocketSend<'a> {
    pub run_id: &'a str,
    pub gateway_instance_id: &'a str,
    pub monotonic_clock_domain_id: &'a str,
    pub client_id: u64,
    pub publication_sequence: u64,
    pub publication_to_selection_ms: f64,
    /// `None` for the initial snapshot, which was selected before a watch wake.
    pub watch_wakeup_to_consume_ms: Option<f64>,
    pub message_assembly_ms: f64,
    pub json_serialization_ms: f64,
    pub socket_send_call_ms: Option<f64>,
    pub payload_bytes: Option<usize>,
    pub outcome: &'static str,
    pub repeated_publication: bool,
    pub repeated_source_observation: bool,
    pub publication_sequence_gap: u64,
    pub source_sequence_gap: Option<u64>,
    pub engine_instance_id: Option<&'a str>,
    pub source_session_id: Option<&'a str>,
    pub source_frame_id: Option<&'a str>,
    pub transport_sequence: Option<u64>,
    pub snapshot_revision: Option<u64>,
    pub configuration_revision: Option<u64>,
    pub rectification_revision: Option<u64>,
    pub calibration_revision: Option<u64>,
    pub freshness: Option<&'a str>,
    pub ball_positions_status: Option<&'a str>,
}

/// Final producer-side counters attached to the graceful audit footer.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LiveAuditShutdownSummary {
    /// True only when session processing and capture-source teardown both completed.
    pub clean_shutdown: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_counters: Option<CapturePipelineStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scene_transport_counters: Option<SceneTransportStats>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LiveAuditExportSummary {
    pub connected: bool,
    pub records_written: u64,
    pub dropped_records: u64,
    pub scene_socket_write_successes: u64,
    pub scene_socket_write_failures: u64,
    pub skipped_observations: u64,
    pub rejected_observations: u64,
    pub failed_observations: u64,
    pub gateway_websocket_send_successes: u64,
    pub gateway_websocket_send_deadlines: u64,
    pub gateway_websocket_send_failures: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
struct LiveAuditCounters {
    records_written: u64,
    scene_socket_write_successes: u64,
    scene_socket_write_failures: u64,
    skipped_observations: u64,
    rejected_observations: u64,
    failed_observations: u64,
    gateway_websocket_send_successes: u64,
    gateway_websocket_send_deadlines: u64,
    gateway_websocket_send_failures: u64,
}

struct RecorderState {
    closed: AtomicBool,
    active_enqueues: AtomicUsize,
    dropped_records: AtomicU64,
}

/// Cloneable non-blocking producer handle. `try_send` is the only queue
/// operation used on frame and transport threads.
#[derive(Clone)]
pub struct LiveAuditRecorder {
    sender: SyncSender<AuditCommand>,
    state: Arc<RecorderState>,
}

impl LiveAuditRecorder {
    pub(crate) fn record_frame(&self, frame: LiveAuditFrame<'_>) {
        let record = AuditRecord::Frame {
            schema_version: LIVE_AUDIT_SCHEMA_VERSION,
            process_id: std::process::id(),
            engine_instance_id: frame.engine_instance_id.to_owned(),
            source_session_id: frame.source_session_id.to_owned(),
            frame_id: frame.frame_id.to_owned(),
            transport_sequence: frame.transport_sequence,
            configuration_revision: frame.configuration_revision,
            tracking_configuration_revision: frame.tracking_configuration_revision,
            rectification_revision: frame.rectification_revision,
            calibration_revision: frame.calibration_revision,
            detection_count: frame.detection_count,
            track_count: frame.track_count,
            source_timestamp: frame.source_timestamp.cloned(),
            frame_timing: frame.frame_timing.cloned(),
            durations_ms: FrameDurationsMs {
                capture_helper_appsink_pull_wait: frame
                    .capture_timing
                    .helper
                    .map(|timing| ns_to_ms(timing.appsink_pull_wait_ns)),
                capture_helper_rgb_buffer_allocation: frame
                    .capture_timing
                    .helper
                    .map(|timing| ns_to_ms(timing.rgb_buffer_allocation_ns)),
                capture_helper_rgb_mapping_copy: frame
                    .capture_timing
                    .helper
                    .map(|timing| ns_to_ms(timing.rgb_mapping_copy_ns)),
                parent_framed_record_read: frame
                    .capture_timing
                    .parent_framed_record_read_ns
                    .map(ns_to_ms),
                capture_receiver_queue_wait: frame
                    .capture_timing
                    .parent_receiver_queue_wait_ns
                    .map(ns_to_ms),
                parent_record_read_complete_to_inference_start: frame
                    .parent_record_read_complete_to_inference_start_ms,
                preprocess: frame.preprocess_ms,
                inference: frame.inference_ms,
                decode_and_nms: frame.decode_and_nms_ms,
                engine_total: frame.engine_total_ms,
                calibration: frame.calibration_ms,
                tracking_and_state_assembly: frame.tracking_state_ms,
                scene_json_serialization: frame.scene_serialization_ms,
                scene_queue_submit: frame.scene_queue_submit_ms,
                parent_record_read_complete_to_scene_queue_submit: frame
                    .parent_record_read_complete_to_scene_queue_ms,
            },
        };
        self.try_record(record);
    }

    pub(crate) fn record_scene_socket_write(&self, write: LiveAuditSceneWrite<'_>) {
        self.try_record(AuditRecord::SceneSocketWrite {
            schema_version: LIVE_AUDIT_SCHEMA_VERSION,
            process_id: std::process::id(),
            engine_instance_id: write.engine_instance_id.to_owned(),
            source_session_id: write.source_session_id.to_owned(),
            frame_id: write.frame_id.to_owned(),
            transport_sequence: write.transport_sequence,
            write_attempt: write.write_attempt,
            outcome: if write.succeeded { "written" } else { "failed" },
            duration_ms: write.duration_ms,
        });
    }

    pub(crate) fn record_gateway_websocket_send(&self, send: LiveAuditGatewayWebSocketSend<'_>) {
        self.try_record(AuditRecord::GatewayWebSocketSend {
            schema_version: LIVE_AUDIT_SCHEMA_VERSION,
            process_id: std::process::id(),
            run_id: send.run_id.to_owned(),
            gateway_instance_id: send.gateway_instance_id.to_owned(),
            monotonic_clock_domain_id: send.monotonic_clock_domain_id.to_owned(),
            client_id: send.client_id,
            publication_sequence: send.publication_sequence,
            publication_to_selection_ms: send.publication_to_selection_ms,
            watch_wakeup_to_consume_ms: send.watch_wakeup_to_consume_ms,
            message_assembly_ms: send.message_assembly_ms,
            json_serialization_ms: send.json_serialization_ms,
            socket_send_call_ms: send.socket_send_call_ms,
            payload_bytes: send.payload_bytes,
            outcome: send.outcome,
            repeated_publication: send.repeated_publication,
            repeated_source_observation: send.repeated_source_observation,
            publication_sequence_gap: send.publication_sequence_gap,
            source_sequence_gap: send.source_sequence_gap,
            engine_instance_id: send.engine_instance_id.map(str::to_owned),
            source_session_id: send.source_session_id.map(str::to_owned),
            source_frame_id: send.source_frame_id.map(str::to_owned),
            transport_sequence: send.transport_sequence,
            snapshot_revision: send.snapshot_revision,
            configuration_revision: send.configuration_revision,
            rectification_revision: send.rectification_revision,
            calibration_revision: send.calibration_revision,
            freshness: send.freshness.map(str::to_owned),
            ball_positions_status: send.ball_positions_status.map(str::to_owned),
        });
    }

    pub(crate) fn record_observation_outcome(&self, observation: LiveAuditObservationOutcome<'_>) {
        self.try_record(AuditRecord::ObservationOutcome {
            schema_version: LIVE_AUDIT_SCHEMA_VERSION,
            process_id: std::process::id(),
            engine_instance_id: observation.engine_instance_id.to_owned(),
            source_session_id: observation.source_session_id.to_owned(),
            frame_id: observation.frame_id.to_owned(),
            transport_sequence: observation.transport_sequence,
            configuration_revision: observation.configuration_revision,
            tracking_configuration_revision: observation.tracking_configuration_revision,
            rectification_revision: observation.rectification_revision,
            calibration_revision: observation.calibration_revision,
            source_timestamp: observation.source_timestamp.cloned(),
            outcome: observation.outcome,
            stage: observation.stage,
            reason: observation.reason.to_owned(),
            inference_timings_ms: observation.inference_timings.map(|timings| {
                InferenceOutcomeDurationsMs {
                    capture_helper_appsink_pull_wait: timings
                        .capture_timing
                        .helper
                        .map(|timing| ns_to_ms(timing.appsink_pull_wait_ns)),
                    capture_helper_rgb_buffer_allocation: timings
                        .capture_timing
                        .helper
                        .map(|timing| ns_to_ms(timing.rgb_buffer_allocation_ns)),
                    capture_helper_rgb_mapping_copy: timings
                        .capture_timing
                        .helper
                        .map(|timing| ns_to_ms(timing.rgb_mapping_copy_ns)),
                    parent_framed_record_read: timings
                        .capture_timing
                        .parent_framed_record_read_ns
                        .map(ns_to_ms),
                    capture_receiver_queue_wait: timings
                        .capture_timing
                        .parent_receiver_queue_wait_ns
                        .map(ns_to_ms),
                    parent_record_read_complete_to_inference_start: timings
                        .parent_record_read_complete_to_inference_start_ms,
                    preprocess: timings.inference.preprocess_ms,
                    inference: timings.inference.inference_ms,
                    decode_and_nms: timings.inference.decode_and_nms_ms,
                    engine_total: timings.inference.engine_total_ms,
                    parent_record_read_complete_to_inference_complete: timings
                        .parent_record_read_complete_to_inference_complete_ms,
                    maximum_frame_age: timings.maximum_frame_age_ms,
                }
            }),
        });
    }

    fn try_record(&self, record: AuditRecord) {
        // The short active-enqueue guard ensures `finish` cannot write its
        // footer while a producer that observed the open flag is still trying
        // to place a record. Producer work never waits for the worker or lock.
        self.state.active_enqueues.fetch_add(1, Ordering::AcqRel);
        if self.state.closed.load(Ordering::Acquire) {
            self.state.active_enqueues.fetch_sub(1, Ordering::Release);
            self.state.dropped_records.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let result = self.sender.try_send(AuditCommand::Record(record));
        if matches!(
            result,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_))
        ) {
            self.state.dropped_records.fetch_add(1, Ordering::Relaxed);
        }
        self.state.active_enqueues.fetch_sub(1, Ordering::Release);
    }
}

/// Owns the sole audit worker and emits a final footer when the receiver is
/// reachable. Dropping it drains bounded queued events and joins the worker.
pub struct LiveAuditExporter {
    recorder: Option<LiveAuditRecorder>,
    worker: Option<JoinHandle<LiveAuditExportSummary>>,
}

impl LiveAuditExporter {
    pub fn start(config: LiveAuditConfig) -> Result<Self> {
        config.validate()?;
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let state = Arc::new(RecorderState {
            closed: AtomicBool::new(false),
            active_enqueues: AtomicUsize::new(0),
            dropped_records: AtomicU64::new(0),
        });
        let recorder = LiveAuditRecorder { sender, state };
        let worker_recorder = recorder.clone();
        let worker = thread::Builder::new()
            .name("live-pipeline-audit-writer".to_owned())
            .spawn(move || run_audit_worker(receiver, worker_recorder, config))
            .context("failed to start live audit writer thread")?;
        Ok(Self {
            recorder: Some(recorder),
            worker: Some(worker),
        })
    }

    pub fn recorder(&self) -> LiveAuditRecorder {
        self.recorder
            .as_ref()
            .expect("live audit exporter recorder exists until shutdown")
            .clone()
    }

    /// Close producers, enqueue a shutdown marker after their in-flight
    /// non-blocking submissions, then wait for the bounded exporter to finish.
    pub fn finish(mut self) -> LiveAuditExportSummary {
        self.finish_inner(LiveAuditShutdownSummary::default())
    }

    pub fn finish_with_shutdown_summary(
        mut self,
        final_summary: LiveAuditShutdownSummary,
    ) -> LiveAuditExportSummary {
        self.finish_inner(final_summary)
    }

    fn finish_inner(&mut self, final_summary: LiveAuditShutdownSummary) -> LiveAuditExportSummary {
        if let Some(recorder) = self.recorder.take() {
            recorder.state.closed.store(true, Ordering::Release);
            while recorder.state.active_enqueues.load(Ordering::Acquire) != 0 {
                thread::yield_now();
            }
            let _ = recorder.sender.send(AuditCommand::Shutdown(final_summary));
        }
        self.worker
            .take()
            .and_then(|worker| worker.join().ok())
            .unwrap_or_else(|| LiveAuditExportSummary {
                failure: Some("live audit writer thread panicked".to_owned()),
                ..LiveAuditExportSummary::default()
            })
    }
}

impl Drop for LiveAuditExporter {
    fn drop(&mut self) {
        if self.worker.is_some() {
            let summary = self.finish_inner(LiveAuditShutdownSummary::default());
            if summary.failure.is_some() || summary.dropped_records > 0 {
                eprintln!(
                    "live_audit_export_summary={}",
                    serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_owned())
                );
            }
        }
    }
}

// Keep audit records inline when entering the bounded queue so producing one
// record does not require an additional allocation; queue capacity bounds the
// retained memory and oversized records are rejected before enqueue.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum AuditCommand {
    Record(AuditRecord),
    Shutdown(LiveAuditShutdownSummary),
}

#[derive(Debug, Serialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
enum AuditRecord {
    Frame {
        schema_version: u8,
        process_id: u32,
        engine_instance_id: String,
        source_session_id: String,
        frame_id: String,
        transport_sequence: u64,
        configuration_revision: u64,
        tracking_configuration_revision: u64,
        rectification_revision: u64,
        calibration_revision: Option<u64>,
        detection_count: usize,
        track_count: usize,
        /// Null means the source exposed no trustworthy timestamp.
        source_timestamp: Option<BatchTimestamp>,
        /// Bounded nested spans on the same process monotonic clock; null means
        /// the timing collector could not produce a complete frame snapshot.
        frame_timing: Option<FrameTimingSnapshot>,
        durations_ms: FrameDurationsMs,
    },
    SceneSocketWrite {
        schema_version: u8,
        process_id: u32,
        engine_instance_id: String,
        source_session_id: String,
        frame_id: String,
        transport_sequence: u64,
        write_attempt: u64,
        outcome: &'static str,
        duration_ms: f64,
    },
    ObservationOutcome {
        schema_version: u8,
        process_id: u32,
        engine_instance_id: String,
        source_session_id: String,
        frame_id: String,
        transport_sequence: Option<u64>,
        configuration_revision: u64,
        tracking_configuration_revision: u64,
        rectification_revision: u64,
        calibration_revision: Option<u64>,
        /// Null means the source supplied no timestamp; this event does not
        /// invent a sensor or exposure timestamp when one is unavailable.
        source_timestamp: Option<BatchTimestamp>,
        outcome: &'static str,
        stage: &'static str,
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        inference_timings_ms: Option<InferenceOutcomeDurationsMs>,
    },
    #[serde(rename = "gateway_websocket_send")]
    GatewayWebSocketSend {
        schema_version: u8,
        process_id: u32,
        run_id: String,
        gateway_instance_id: String,
        monotonic_clock_domain_id: String,
        client_id: u64,
        publication_sequence: u64,
        publication_to_selection_ms: f64,
        watch_wakeup_to_consume_ms: Option<f64>,
        message_assembly_ms: f64,
        json_serialization_ms: f64,
        socket_send_call_ms: Option<f64>,
        payload_bytes: Option<usize>,
        outcome: &'static str,
        repeated_publication: bool,
        repeated_source_observation: bool,
        publication_sequence_gap: u64,
        source_sequence_gap: Option<u64>,
        engine_instance_id: Option<String>,
        source_session_id: Option<String>,
        source_frame_id: Option<String>,
        transport_sequence: Option<u64>,
        snapshot_revision: Option<u64>,
        configuration_revision: Option<u64>,
        rectification_revision: Option<u64>,
        calibration_revision: Option<u64>,
        freshness: Option<String>,
        ball_positions_status: Option<String>,
    },
}

#[derive(Debug, Serialize)]
struct FrameDurationsMs {
    /// Unavailable for sources that do not expose helper capture timings.
    capture_helper_appsink_pull_wait: Option<f64>,
    capture_helper_rgb_buffer_allocation: Option<f64>,
    capture_helper_rgb_mapping_copy: Option<f64>,
    /// Parent-local duration covering the full framed record read and RGB copy.
    parent_framed_record_read: Option<f64>,
    /// Parent-local frame completion through capacity-one receiver dequeue.
    capture_receiver_queue_wait: Option<f64>,
    /// Parent record read completion through inference start, including receiver wait.
    parent_record_read_complete_to_inference_start: f64,
    preprocess: f64,
    inference: f64,
    decode_and_nms: f64,
    engine_total: f64,
    /// Null means the runtime had no valid table calibration.
    calibration: Option<f64>,
    tracking_and_state_assembly: Option<f64>,
    scene_json_serialization: Option<f64>,
    scene_queue_submit: Option<f64>,
    /// Parent record read completion through successful scene queue submission.
    parent_record_read_complete_to_scene_queue_submit: f64,
}

#[derive(Debug, Serialize)]
struct InferenceOutcomeDurationsMs {
    /// Helper-process capture spans; unavailable if the source did not expose them.
    capture_helper_appsink_pull_wait: Option<f64>,
    capture_helper_rgb_buffer_allocation: Option<f64>,
    capture_helper_rgb_mapping_copy: Option<f64>,
    /// Parent-local span for reading and copying the complete framed record.
    parent_framed_record_read: Option<f64>,
    /// Parent-local frame completion through capacity-one receiver dequeue.
    capture_receiver_queue_wait: Option<f64>,
    /// Includes receiver queue delay after the parent completed the framed read.
    parent_record_read_complete_to_inference_start: f64,
    preprocess: f64,
    inference: f64,
    decode_and_nms: f64,
    engine_total: f64,
    /// Freshness age measured in the parent process's monotonic clock domain.
    parent_record_read_complete_to_inference_complete: f64,
    maximum_frame_age: f64,
}

fn ns_to_ms(value: u64) -> f64 {
    value as f64 / 1_000_000.0
}

#[derive(Debug, Serialize)]
struct AuditFooter {
    record_type: &'static str,
    schema_version: u8,
    export_status: &'static str,
    records_written_before_footer: u64,
    dropped_records: u64,
    scene_socket_write_successes: u64,
    scene_socket_write_failures: u64,
    skipped_observations: u64,
    rejected_observations: u64,
    failed_observations: u64,
    gateway_websocket_send_successes: u64,
    gateway_websocket_send_deadlines: u64,
    gateway_websocket_send_failures: u64,
    clean_shutdown: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pipeline_counters: Option<CapturePipelineStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scene_transport_counters: Option<SceneTransportStats>,
}

fn run_audit_worker(
    receiver: Receiver<AuditCommand>,
    recorder: LiveAuditRecorder,
    config: LiveAuditConfig,
) -> LiveAuditExportSummary {
    let mut stream = match TcpStream::connect_timeout(&config.address, config.connect_timeout) {
        Ok(stream) => stream,
        Err(error) => {
            let dropped = discard_until_shutdown(&receiver, &recorder);
            return LiveAuditExportSummary {
                dropped_records: dropped,
                failure: Some(format!("audit receiver connection failed: {error}")),
                ..LiveAuditExportSummary::default()
            };
        }
    };
    if let Err(error) = stream.set_nodelay(true) {
        let dropped = discard_until_shutdown(&receiver, &recorder);
        return LiveAuditExportSummary {
            connected: true,
            dropped_records: dropped,
            failure: Some(format!("failed to enable audit TCP_NODELAY: {error}")),
            ..LiveAuditExportSummary::default()
        };
    }

    let mut counters = LiveAuditCounters::default();
    while let Ok(command) = receiver.recv() {
        match command {
            AuditCommand::Shutdown(final_summary) => {
                let footer = AuditFooter {
                    record_type: "footer",
                    schema_version: LIVE_AUDIT_SCHEMA_VERSION,
                    export_status: "complete",
                    records_written_before_footer: counters.records_written,
                    dropped_records: recorder.state.dropped_records.load(Ordering::Relaxed),
                    scene_socket_write_successes: counters.scene_socket_write_successes,
                    scene_socket_write_failures: counters.scene_socket_write_failures,
                    skipped_observations: counters.skipped_observations,
                    rejected_observations: counters.rejected_observations,
                    failed_observations: counters.failed_observations,
                    gateway_websocket_send_successes: counters.gateway_websocket_send_successes,
                    gateway_websocket_send_deadlines: counters.gateway_websocket_send_deadlines,
                    gateway_websocket_send_failures: counters.gateway_websocket_send_failures,
                    clean_shutdown: final_summary.clean_shutdown,
                    pipeline_counters: final_summary.pipeline_counters,
                    scene_transport_counters: final_summary.scene_transport_counters,
                };
                let footer_bytes = match serde_json::to_vec(&footer) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        return summary_from(
                            true,
                            counters,
                            recorder.state.dropped_records.load(Ordering::Relaxed),
                            Some(format!("failed to serialize audit footer: {error}")),
                        );
                    }
                };
                if let Err(error) = write_record(
                    &mut stream,
                    &footer_bytes,
                    config.maximum_record_bytes,
                    config.write_timeout,
                ) {
                    return summary_from(
                        true,
                        counters,
                        recorder.state.dropped_records.load(Ordering::Relaxed),
                        Some(format!("failed to write audit footer: {error}")),
                    );
                }
                counters.records_written = counters.records_written.saturating_add(1);
                return summary_from(
                    true,
                    counters,
                    recorder.state.dropped_records.load(Ordering::Relaxed),
                    None,
                );
            }
            AuditCommand::Record(record) => {
                let transport_write = match &record {
                    AuditRecord::SceneSocketWrite { outcome, .. } => Some(*outcome == "written"),
                    _ => None,
                };
                let observation_outcome = match &record {
                    AuditRecord::ObservationOutcome { outcome, .. } => Some(*outcome),
                    _ => None,
                };
                let gateway_send_outcome = match &record {
                    AuditRecord::GatewayWebSocketSend { outcome, .. } => Some(*outcome),
                    _ => None,
                };
                let bytes = match serde_json::to_vec(&record) {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        recorder
                            .state
                            .dropped_records
                            .fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                };
                if bytes.len() > config.maximum_record_bytes {
                    recorder
                        .state
                        .dropped_records
                        .fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                if let Err(error) = write_record(
                    &mut stream,
                    &bytes,
                    config.maximum_record_bytes,
                    config.write_timeout,
                ) {
                    // This record has already left the bounded queue but did
                    // not reach the receiver. Count it before draining queued
                    // records so partial-export summaries include every loss.
                    recorder
                        .state
                        .dropped_records
                        .fetch_add(1, Ordering::Relaxed);
                    let dropped = discard_until_shutdown(&receiver, &recorder);
                    return summary_from(
                        true,
                        counters,
                        dropped,
                        Some(format!("audit record write failed: {error}")),
                    );
                }
                counters.records_written = counters.records_written.saturating_add(1);
                match transport_write {
                    Some(true) => {
                        counters.scene_socket_write_successes =
                            counters.scene_socket_write_successes.saturating_add(1);
                    }
                    Some(false) => {
                        counters.scene_socket_write_failures =
                            counters.scene_socket_write_failures.saturating_add(1);
                    }
                    None => {}
                }
                match observation_outcome {
                    Some("skipped") => {
                        counters.skipped_observations =
                            counters.skipped_observations.saturating_add(1);
                    }
                    Some("rejected") => {
                        counters.rejected_observations =
                            counters.rejected_observations.saturating_add(1);
                    }
                    Some("failed") => {
                        counters.failed_observations =
                            counters.failed_observations.saturating_add(1);
                    }
                    _ => {}
                }
                match gateway_send_outcome {
                    Some("sent") => {
                        counters.gateway_websocket_send_successes =
                            counters.gateway_websocket_send_successes.saturating_add(1);
                    }
                    Some("deadline_exceeded") => {
                        counters.gateway_websocket_send_deadlines =
                            counters.gateway_websocket_send_deadlines.saturating_add(1);
                    }
                    Some(_) => {
                        counters.gateway_websocket_send_failures =
                            counters.gateway_websocket_send_failures.saturating_add(1);
                    }
                    None => {}
                }
            }
        }
    }
    summary_from(
        true,
        counters,
        recorder.state.dropped_records.load(Ordering::Relaxed),
        Some("live audit queue disconnected before shutdown footer".to_owned()),
    )
}

fn discard_until_shutdown(receiver: &Receiver<AuditCommand>, recorder: &LiveAuditRecorder) -> u64 {
    let mut discarded = 0_u64;
    while let Ok(command) = receiver.recv() {
        match command {
            AuditCommand::Shutdown(_) => break,
            AuditCommand::Record(_) => discarded = discarded.saturating_add(1),
        }
    }
    recorder
        .state
        .dropped_records
        .load(Ordering::Relaxed)
        .saturating_add(discarded)
}

fn summary_from(
    connected: bool,
    counters: LiveAuditCounters,
    dropped_records: u64,
    failure: Option<String>,
) -> LiveAuditExportSummary {
    LiveAuditExportSummary {
        connected,
        records_written: counters.records_written,
        dropped_records,
        scene_socket_write_successes: counters.scene_socket_write_successes,
        scene_socket_write_failures: counters.scene_socket_write_failures,
        skipped_observations: counters.skipped_observations,
        rejected_observations: counters.rejected_observations,
        failed_observations: counters.failed_observations,
        gateway_websocket_send_successes: counters.gateway_websocket_send_successes,
        gateway_websocket_send_deadlines: counters.gateway_websocket_send_deadlines,
        gateway_websocket_send_failures: counters.gateway_websocket_send_failures,
        failure,
    }
}

fn write_record(
    stream: &mut TcpStream,
    payload: &[u8],
    maximum_record_bytes: usize,
    timeout: Duration,
) -> io::Result<()> {
    if payload.is_empty()
        || payload.len() > maximum_record_bytes
        || payload.len() > u32::MAX as usize
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "live audit record is empty or exceeds its configured frame bound",
        ));
    }
    let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "live audit write deadline cannot be represented",
        )
    })?;
    let header = (payload.len() as u32).to_be_bytes();
    write_all_until(stream, &header, deadline)?;
    write_all_until(stream, payload, deadline)
}

fn write_all_until(
    stream: &mut TcpStream,
    mut remaining: &[u8],
    deadline: Instant,
) -> io::Result<()> {
    while !remaining.is_empty() {
        let duration = deadline.saturating_duration_since(Instant::now());
        if duration.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "live audit socket write exceeded its configured deadline",
            ));
        }
        stream.set_write_timeout(Some(duration))?;
        match stream.write(remaining) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "live audit socket closed during a record write",
                ));
            }
            Ok(written) => remaining = &remaining[written..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, TcpListener};
    use std::thread;

    fn config(address: SocketAddr) -> LiveAuditConfig {
        LiveAuditConfig {
            address,
            queue_capacity: 4,
            maximum_record_bytes: 16 * 1024,
            connect_timeout: Duration::from_secs(1),
            write_timeout: Duration::from_secs(1),
        }
    }

    fn localhost_address() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
    }

    #[test]
    fn configuration_rejects_remote_and_unbounded_settings() {
        let mut invalid = config(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9));
        assert!(invalid.validate().is_err());

        invalid = config(localhost_address());
        invalid.queue_capacity = 0;
        assert!(invalid.validate().is_err());
        invalid.queue_capacity = MAX_LIVE_AUDIT_QUEUE_CAPACITY + 1;
        assert!(invalid.validate().is_err());

        invalid = config(localhost_address());
        invalid.maximum_record_bytes = MIN_LIVE_AUDIT_RECORD_BYTES - 1;
        assert!(invalid.validate().is_err());
        invalid.maximum_record_bytes = MAX_LIVE_AUDIT_RECORD_BYTES + 1;
        assert!(invalid.validate().is_err());

        invalid = config(localhost_address());
        invalid.connect_timeout = Duration::ZERO;
        assert!(invalid.validate().is_err());
        invalid.connect_timeout = Duration::from_millis(1);
        invalid.write_timeout = Duration::ZERO;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn frame_and_footer_records_are_correlated_and_contain_no_scene_data() {
        let listener = TcpListener::bind(localhost_address()).unwrap();
        let address = listener.local_addr().unwrap();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let frame: serde_json::Value =
                serde_json::from_slice(&read_record(&mut stream)).unwrap();
            let transport: serde_json::Value =
                serde_json::from_slice(&read_record(&mut stream)).unwrap();
            let footer: serde_json::Value =
                serde_json::from_slice(&read_record(&mut stream)).unwrap();
            (frame, transport, footer)
        });

        let exporter = LiveAuditExporter::start(config(address)).unwrap();
        let recorder = exporter.recorder();
        let timestamp = BatchTimestamp {
            ticks: "123456".to_owned(),
            unit: crate::detection_batch::TimeUnit::Nanoseconds,
            clock: crate::detection_batch::ClockKind::Monotonic,
            domain_id: "engine-a".to_owned(),
            stage: crate::detection_batch::TimestampStage::CameraReadComplete,
        };
        recorder.record_frame(LiveAuditFrame {
            engine_instance_id: "engine-a",
            source_session_id: "session-b",
            frame_id: "frame-c",
            transport_sequence: 7,
            configuration_revision: 5,
            tracking_configuration_revision: 6,
            rectification_revision: 3,
            calibration_revision: None,
            detection_count: 2,
            track_count: 1,
            source_timestamp: Some(&timestamp),
            capture_timing: crate::capture::CaptureFrameTiming {
                helper: Some(crate::capture::CaptureHelperTimingNs {
                    appsink_pull_wait_ns: 10_000,
                    rgb_buffer_allocation_ns: 20_000,
                    rgb_mapping_copy_ns: 30_000,
                }),
                parent_framed_record_read_ns: Some(40_000),
                parent_receiver_queue_wait_ns: Some(50_000),
            },
            parent_record_read_complete_to_inference_start_ms: 0.4,
            preprocess_ms: 1.2,
            inference_ms: 3.4,
            decode_and_nms_ms: 0.6,
            engine_total_ms: 5.7,
            calibration_ms: None,
            tracking_state_ms: Some(0.2),
            scene_serialization_ms: Some(0.1),
            scene_queue_submit_ms: Some(0.03),
            parent_record_read_complete_to_scene_queue_ms: 5.9,
            frame_timing: None,
        });
        recorder.record_scene_socket_write(LiveAuditSceneWrite {
            engine_instance_id: "engine-a",
            source_session_id: "session-b",
            frame_id: "frame-c",
            transport_sequence: 7,
            write_attempt: 1,
            duration_ms: 0.05,
            succeeded: true,
        });
        drop(recorder);
        let summary = exporter.finish();
        let (frame, transport, footer) = receiver.join().unwrap();

        assert_eq!(summary.failure, None);
        assert!(summary.connected);
        assert_eq!(summary.scene_socket_write_successes, 1);
        assert_eq!(frame["record_type"], "frame");
        assert_eq!(frame["schema_version"], LIVE_AUDIT_SCHEMA_VERSION);
        assert_eq!(frame["engine_instance_id"], "engine-a");
        assert_eq!(frame["source_session_id"], "session-b");
        assert_eq!(frame["frame_id"], "frame-c");
        assert_eq!(frame["transport_sequence"], 7);
        assert_eq!(frame["configuration_revision"], 5);
        assert_eq!(frame["tracking_configuration_revision"], 6);
        assert_eq!(frame["rectification_revision"], 3);
        assert_eq!(frame["calibration_revision"], serde_json::Value::Null);
        assert_eq!(frame["detection_count"], 2);
        assert_eq!(frame["track_count"], 1);
        assert_eq!(frame["source_timestamp"]["stage"], "camera_read_complete");
        assert_eq!(
            frame["durations_ms"]["capture_helper_appsink_pull_wait"],
            0.01
        );
        assert_eq!(
            frame["durations_ms"]["capture_helper_rgb_buffer_allocation"],
            0.02
        );
        assert_eq!(
            frame["durations_ms"]["capture_helper_rgb_mapping_copy"],
            0.03
        );
        assert_eq!(frame["durations_ms"]["parent_framed_record_read"], 0.04);
        assert_eq!(frame["durations_ms"]["capture_receiver_queue_wait"], 0.05);
        assert_eq!(frame["durations_ms"]["engine_total"], 5.7);
        assert_eq!(
            frame["durations_ms"]["calibration"],
            serde_json::Value::Null
        );
        assert!(frame.get("balls").is_none());
        assert!(frame.get("detections").is_none());
        assert!(frame.get("rgb").is_none());
        assert_eq!(transport["record_type"], "scene_socket_write");
        assert_eq!(transport["engine_instance_id"], "engine-a");
        assert_eq!(transport["source_session_id"], "session-b");
        assert_eq!(transport["frame_id"], "frame-c");
        assert_eq!(transport["transport_sequence"], 7);
        assert_eq!(transport["outcome"], "written");
        assert_eq!(footer["record_type"], "footer");
        assert_eq!(footer["export_status"], "complete");
        assert_eq!(footer["dropped_records"], 0);
    }

    #[test]
    fn gateway_websocket_record_keeps_publication_identity_and_send_outcome() {
        let listener = TcpListener::bind(localhost_address()).unwrap();
        let address = listener.local_addr().unwrap();
        let receiver = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let send: serde_json::Value =
                serde_json::from_slice(&read_record(&mut stream)).unwrap();
            let footer: serde_json::Value =
                serde_json::from_slice(&read_record(&mut stream)).unwrap();
            (send, footer)
        });

        let exporter = LiveAuditExporter::start(config(address)).unwrap();
        exporter
            .recorder()
            .record_gateway_websocket_send(LiveAuditGatewayWebSocketSend {
                run_id: "run-a",
                gateway_instance_id: "gateway-a",
                monotonic_clock_domain_id: "gateway-clock-a",
                client_id: 17,
                publication_sequence: 22,
                publication_to_selection_ms: 1.5,
                watch_wakeup_to_consume_ms: Some(0.01),
                message_assembly_ms: 0.2,
                json_serialization_ms: 0.3,
                socket_send_call_ms: Some(2.0),
                payload_bytes: Some(4096),
                outcome: "sent",
                repeated_publication: false,
                repeated_source_observation: false,
                publication_sequence_gap: 2,
                source_sequence_gap: Some(1),
                engine_instance_id: Some("engine-a"),
                source_session_id: Some("session-b"),
                source_frame_id: Some("frame-c"),
                transport_sequence: Some(12),
                snapshot_revision: Some(9),
                configuration_revision: Some(4),
                rectification_revision: Some(2),
                calibration_revision: Some(3),
                freshness: Some("fresh"),
                ball_positions_status: Some("observed"),
            });
        let summary = exporter.finish();
        let (send, footer) = receiver.join().unwrap();

        assert_eq!(summary.gateway_websocket_send_successes, 1);
        assert_eq!(send["record_type"], "gateway_websocket_send");
        assert_eq!(send["run_id"], "run-a");
        assert_eq!(send["process_id"], std::process::id());
        assert_eq!(send["client_id"], 17);
        assert_eq!(send["publication_sequence"], 22);
        assert_eq!(send["publication_sequence_gap"], 2);
        assert_eq!(send["source_sequence_gap"], 1);
        assert_eq!(send["source_frame_id"], "frame-c");
        assert_eq!(send["socket_send_call_ms"], 2.0);
        assert_eq!(send["payload_bytes"], 4096);
        assert_eq!(send["outcome"], "sent");
        assert!(send.get("balls").is_none());
        assert_eq!(footer["gateway_websocket_send_successes"], 1);
        assert_eq!(footer["gateway_websocket_send_deadlines"], 0);
    }

    #[test]
    fn bounded_enqueue_reports_full_queue_without_waiting_for_consumer() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        let state = Arc::new(RecorderState {
            closed: AtomicBool::new(false),
            active_enqueues: AtomicUsize::new(0),
            dropped_records: AtomicU64::new(0),
        });
        let recorder = LiveAuditRecorder { sender, state };
        let make_record = || AuditRecord::SceneSocketWrite {
            schema_version: LIVE_AUDIT_SCHEMA_VERSION,
            process_id: 1,
            engine_instance_id: "engine".to_owned(),
            source_session_id: "session".to_owned(),
            frame_id: "frame".to_owned(),
            transport_sequence: 1,
            write_attempt: 1,
            outcome: "written",
            duration_ms: 1.0,
        };

        let started = Instant::now();
        recorder.try_record(make_record());
        recorder.try_record(make_record());
        assert!(started.elapsed() < Duration::from_millis(100));
        assert_eq!(recorder.state.dropped_records.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_receiver_is_audit_only_and_is_reported_as_partial_export() {
        let listener = TcpListener::bind(localhost_address()).unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);

        let exporter = LiveAuditExporter::start(config(address)).unwrap();
        exporter.recorder().record_frame(LiveAuditFrame {
            engine_instance_id: "engine",
            source_session_id: "session",
            frame_id: "frame",
            transport_sequence: 1,
            configuration_revision: 1,
            tracking_configuration_revision: 1,
            rectification_revision: 0,
            calibration_revision: None,
            detection_count: 0,
            track_count: 0,
            source_timestamp: Some(&BatchTimestamp {
                ticks: "1".to_owned(),
                unit: crate::detection_batch::TimeUnit::Nanoseconds,
                clock: crate::detection_batch::ClockKind::Monotonic,
                domain_id: "engine".to_owned(),
                stage: crate::detection_batch::TimestampStage::CameraReadComplete,
            }),
            capture_timing: crate::capture::CaptureFrameTiming::default(),
            parent_record_read_complete_to_inference_start_ms: 0.0,
            preprocess_ms: 0.0,
            inference_ms: 0.0,
            decode_and_nms_ms: 0.0,
            engine_total_ms: 0.0,
            calibration_ms: None,
            tracking_state_ms: None,
            scene_serialization_ms: None,
            scene_queue_submit_ms: None,
            parent_record_read_complete_to_scene_queue_ms: 0.0,
            frame_timing: None,
        });
        let summary = exporter.finish();
        assert!(!summary.connected);
        assert!(summary.failure.is_some());
    }

    fn read_record(stream: &mut TcpStream) -> Vec<u8> {
        use std::io::Read;
        let mut header = [0_u8; 4];
        stream.read_exact(&mut header).unwrap();
        let mut payload = vec![0; u32::from_be_bytes(header) as usize];
        stream.read_exact(&mut payload).unwrap();
        payload
    }
}
