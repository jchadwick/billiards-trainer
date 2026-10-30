//! Repeatable run orchestration and artifact writing for state-stream audits.
//!
//! The receiver measures one WebSocket window. This module owns the separate
//! audit-workflow contract: warm-up, repeated measured windows or one soak,
//! per-run detail files, environment provenance, and a versioned Markdown and
//! JSON report. It labels network-stream input content and detector accuracy as
//! unavailable; when it owns a live server, it also records hashes from that
//! server's effective configuration and model/runtime files.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::live_timing_accumulator::RustLiveTimingAccumulator;
use crate::managed_live_audit::{
    ManagedLiveAuditReceiver, ManagedLiveAuditReceiverConfig, ManagedLiveAuditSnapshot,
};
use crate::performance_audit_receiver::{
    connect_and_collect, connect_and_wait_for_first_fresh_observed_localization,
    validate_websocket_url, StateReceiverConfig, StateReceiverOutput, StateReceiverReport,
};

pub const DEFAULT_WARMUP_SECONDS: u64 = 30;
pub const DEFAULT_MEASURED_SECONDS: u64 = 120;
pub const DEFAULT_REPETITIONS: usize = 3;
pub const DEFAULT_SOAK_SECONDS: u64 = 1_200;
pub const RUN_REPORT_SCHEMA_VERSION: u8 = 8;
pub const DEFAULT_LIVE_AUDIT_QUEUE_CAPACITY: usize = 2_048;
pub const DEFAULT_LIVE_AUDIT_MAX_RECORD_BYTES: usize = 64 * 1024;
pub const DEFAULT_LIVE_AUDIT_RETAINED_RECORDS: usize = 200_000;
pub const DEFAULT_LIVE_AUDIT_RETAINED_BYTES: usize = 128 * 1024 * 1024;
pub const DEFAULT_LIVE_AUDIT_CONNECT_TIMEOUT: Duration = Duration::from_millis(1_000);
pub const DEFAULT_LIVE_AUDIT_WRITE_TIMEOUT: Duration = Duration::from_millis(1_000);
const MAXIMUM_REPETITIONS: usize = 1_000;
const PROVENANCE_HASH_BUFFER_SIZE: usize = 64 * 1024;

/// Keep generated audit reports outside the source tree by default.
pub fn default_output_root() -> PathBuf {
    std::env::temp_dir().join("billiards-performance-audits")
}

/// Settings for one repeatable state delivery audit.
#[derive(Debug, Clone)]
pub struct StateAuditRunConfig {
    pub url: String,
    pub warmup: Duration,
    pub measured: Duration,
    pub repetitions: usize,
    pub soak: bool,
    pub soak_duration: Duration,
    pub output_root: PathBuf,
    pub receiver: StateReceiverConfig,
    /// Optional live-detector stage timing side channel. `None` allocates no
    /// listener, queue, tasks, or detailed timing retention.
    pub live_pipeline_timing_audit: Option<LivePipelineTimingAuditConfig>,
    /// Optional independent timing stream for per-client gateway WebSocket
    /// publication, serialization, and bounded send calls.
    pub gateway_websocket_timing_audit: Option<GatewayWebSocketTimingAuditConfig>,
    /// Automatically sample the detector child whose PID is carried by the
    /// managed live timing stream. This is separate from sampling an
    /// externally supplied PID through `receiver.resource_sample_process_id`.
    pub sample_managed_live_process_resources: bool,
    /// Optional application server child. Its first observed localization
    /// gates warm-up and measurements, and the child is always reaped before
    /// this command exits.
    pub managed_server: Option<ManagedServerConfig>,
}

/// Explicit resource bounds for the optional live-pipeline timing channel.
#[derive(Debug, Clone)]
pub struct LivePipelineTimingAuditConfig {
    pub queue_capacity: usize,
    pub maximum_record_bytes: usize,
    pub maximum_retained_records: usize,
    pub maximum_retained_bytes: usize,
    pub connect_timeout: Duration,
    pub write_timeout: Duration,
}

/// Resource bounds for the gateway's independent WebSocket timing stream.
/// This intentionally has its own configuration and receiver lifecycle so
/// gateway records never affect detector process discovery or timing joins.
#[derive(Debug, Clone)]
pub struct GatewayWebSocketTimingAuditConfig {
    pub queue_capacity: usize,
    pub maximum_record_bytes: usize,
    pub maximum_retained_records: usize,
    pub maximum_retained_bytes: usize,
    pub connect_timeout: Duration,
    pub write_timeout: Duration,
}

impl Default for GatewayWebSocketTimingAuditConfig {
    fn default() -> Self {
        Self {
            queue_capacity: DEFAULT_LIVE_AUDIT_QUEUE_CAPACITY,
            maximum_record_bytes: DEFAULT_LIVE_AUDIT_MAX_RECORD_BYTES,
            maximum_retained_records: DEFAULT_LIVE_AUDIT_RETAINED_RECORDS,
            maximum_retained_bytes: DEFAULT_LIVE_AUDIT_RETAINED_BYTES,
            connect_timeout: DEFAULT_LIVE_AUDIT_CONNECT_TIMEOUT,
            write_timeout: DEFAULT_LIVE_AUDIT_WRITE_TIMEOUT,
        }
    }
}

impl GatewayWebSocketTimingAuditConfig {
    fn validate(&self) -> Result<()> {
        validate_timing_receiver_config(
            self.queue_capacity,
            self.maximum_record_bytes,
            self.maximum_retained_records,
            self.maximum_retained_bytes,
            self.connect_timeout,
            self.write_timeout,
            "gateway timing",
        )
    }
}

impl Default for LivePipelineTimingAuditConfig {
    fn default() -> Self {
        Self {
            queue_capacity: DEFAULT_LIVE_AUDIT_QUEUE_CAPACITY,
            maximum_record_bytes: DEFAULT_LIVE_AUDIT_MAX_RECORD_BYTES,
            maximum_retained_records: DEFAULT_LIVE_AUDIT_RETAINED_RECORDS,
            maximum_retained_bytes: DEFAULT_LIVE_AUDIT_RETAINED_BYTES,
            connect_timeout: DEFAULT_LIVE_AUDIT_CONNECT_TIMEOUT,
            write_timeout: DEFAULT_LIVE_AUDIT_WRITE_TIMEOUT,
        }
    }
}

impl LivePipelineTimingAuditConfig {
    fn validate(&self) -> Result<()> {
        validate_timing_receiver_config(
            self.queue_capacity,
            self.maximum_record_bytes,
            self.maximum_retained_records,
            self.maximum_retained_bytes,
            self.connect_timeout,
            self.write_timeout,
            "live timing",
        )
    }
}

fn validate_timing_receiver_config(
    queue_capacity: usize,
    maximum_record_bytes: usize,
    maximum_retained_records: usize,
    maximum_retained_bytes: usize,
    connect_timeout: Duration,
    write_timeout: Duration,
    label: &str,
) -> Result<()> {
    if connect_timeout.as_millis() == 0 || write_timeout.as_millis() == 0 {
        bail!("{label} producer deadlines must be at least one millisecond");
    }
    if Instant::now().checked_add(connect_timeout).is_none()
        || Instant::now().checked_add(write_timeout).is_none()
    {
        bail!("{label} producer deadlines exceed the platform clock range");
    }
    if connect_timeout.as_millis() > u64::MAX as u128
        || write_timeout.as_millis() > u64::MAX as u128
    {
        bail!("{label} producer deadlines exceed the server CLI limit");
    }
    if !(1..=crate::live_audit::MAX_LIVE_AUDIT_QUEUE_CAPACITY).contains(&queue_capacity) {
        bail!(
            "{label} queue capacity must be between 1 and {}",
            crate::live_audit::MAX_LIVE_AUDIT_QUEUE_CAPACITY
        );
    }
    if !(crate::live_audit::MIN_LIVE_AUDIT_RECORD_BYTES
        ..=crate::live_audit::MAX_LIVE_AUDIT_RECORD_BYTES)
        .contains(&maximum_record_bytes)
    {
        bail!(
            "{label} record limit must be between {} and {} bytes",
            crate::live_audit::MIN_LIVE_AUDIT_RECORD_BYTES,
            crate::live_audit::MAX_LIVE_AUDIT_RECORD_BYTES
        );
    }
    ManagedLiveAuditReceiverConfig {
        address: SocketAddr::from(([127, 0, 0, 1], 0)),
        maximum_record_bytes,
        queue_capacity,
        maximum_retained_records,
        maximum_retained_bytes,
        maximum_connections: 1,
    }
    .validate()
}

/// Launch and lifecycle limits for a locally managed Rust application server.
#[derive(Debug, Clone)]
pub struct ManagedServerConfig {
    pub executable: PathBuf,
    pub arguments: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub startup_timeout: Duration,
    pub readiness_probe_interval: Duration,
    pub shutdown_timeout: Duration,
}

impl ManagedServerConfig {
    fn validate(&self, url: &str) -> Result<()> {
        if self.executable.as_os_str().is_empty() {
            bail!("managed server executable must not be empty");
        }
        if self.startup_timeout.is_zero() {
            bail!("managed server startup timeout must be positive");
        }
        if self.readiness_probe_interval.is_zero() {
            bail!("managed server readiness probe interval must be positive");
        }
        if self.shutdown_timeout.is_zero() {
            bail!("managed server shutdown timeout must be positive");
        }
        if let Some(directory) = &self.working_directory {
            if !directory.is_dir() {
                bail!(
                    "managed server working directory is not a directory: {}",
                    directory.display()
                );
            }
        }
        validate_loopback_websocket_url(url)
    }
}

impl StateAuditRunConfig {
    pub fn validate(&self) -> Result<()> {
        validate_websocket_url(&self.url)?;
        self.receiver.validate()?;
        if let Some(server) = &self.managed_server {
            server.validate(&self.url)?;
            let has_live_audit_arguments = server
                .arguments
                .iter()
                .any(|argument| argument.starts_with("--live-audit-"));
            if has_live_audit_arguments {
                bail!("managed --live-audit-* arguments are reserved for --live-pipeline-timing-audit");
            }
            if server
                .arguments
                .iter()
                .any(|argument| argument.starts_with("--gateway-audit-"))
            {
                bail!("managed --gateway-audit-* arguments are reserved for --gateway-websocket-timing-audit");
            }
        }
        if let Some(timing) = &self.live_pipeline_timing_audit {
            timing.validate()?;
            if !self.managed_server.as_ref().is_some_and(|server| {
                server
                    .arguments
                    .iter()
                    .any(|argument| argument == "--start-live")
            }) {
                bail!("live pipeline timing audit requires a managed server with --start-live");
            }
        }
        if let Some(timing) = &self.gateway_websocket_timing_audit {
            timing.validate()?;
            if !self.managed_server.as_ref().is_some_and(|server| {
                server
                    .arguments
                    .iter()
                    .any(|argument| argument == "--start-live")
            }) {
                bail!("gateway WebSocket timing audit requires a managed server with --start-live");
            }
        }
        if self.sample_managed_live_process_resources {
            if self.live_pipeline_timing_audit.is_none() {
                bail!(
                    "managed live process resource sampling requires --live-pipeline-timing-audit"
                );
            }
            if self.receiver.resource_sample_process_id.is_some() {
                bail!("managed live process resource sampling cannot be combined with an explicit resource process ID");
            }
        }
        if self.measured.is_zero() {
            bail!("measured duration must be positive");
        }
        if self.soak_duration.is_zero() {
            bail!("soak duration must be positive");
        }
        if !self.soak && !(1..=MAXIMUM_REPETITIONS).contains(&self.repetitions) {
            bail!("repetitions must be between 1 and {MAXIMUM_REPETITIONS}");
        }
        if self.output_root.as_os_str().is_empty() {
            bail!("output root must not be empty");
        }
        Ok(())
    }

    fn measurement_count(&self) -> usize {
        if self.soak {
            1
        } else {
            self.repetitions
        }
    }

    fn measurement_duration(&self) -> Duration {
        if self.soak {
            self.soak_duration
        } else {
            self.measured
        }
    }
}

#[derive(Debug, Serialize)]
pub struct StateAuditRunReport {
    pub schema_version: u8,
    pub tool: &'static str,
    pub run_id: String,
    pub mode: &'static str,
    pub started_at_utc: String,
    pub finished_at_utc: String,
    pub configuration: EffectiveRunConfiguration,
    pub environment: RunEnvironment,
    pub input: InputMetadata,
    pub warmup: Option<WarmupSummary>,
    pub measured_runs: Vec<MeasuredRunReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_pipeline_timing: Option<ManagedLivePipelineTimingSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway_websocket_timing: Option<ManagedGatewayWebSocketTimingSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub managed_server: Option<ManagedServerRunMetadata>,
    pub limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ManagedLivePipelineTimingSummary {
    pub status: &'static str,
    pub listener_address: String,
    pub measurement_clock_domain_id: String,
    pub stream_counts: crate::managed_live_audit::ManagedLiveAuditStreamCounts,
    pub producer_footer: Option<serde_json::Value>,
    pub losses: crate::managed_live_audit::ManagedLiveAuditLosses,
    pub retained_record_count: usize,
    pub retained_record_bytes: usize,
    pub maximum_retained_records: usize,
    pub maximum_retained_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct ManagedGatewayWebSocketTimingSummary {
    pub status: &'static str,
    pub run_id: String,
    pub listener_address: String,
    pub measurement_clock_domain_id: String,
    pub stream_counts: crate::managed_live_audit::ManagedLiveAuditStreamCounts,
    pub producer_footer: Option<serde_json::Value>,
    pub footer_complete: bool,
    pub losses: crate::managed_live_audit::ManagedLiveAuditLosses,
    pub retained_record_count: usize,
    pub retained_record_bytes: usize,
    pub maximum_retained_records: usize,
    pub maximum_retained_bytes: usize,
    /// Gateway process IDs are recorded for provenance only and are never
    /// used as detector resource-sampling targets.
    pub producer_process_id: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct ManagedServerRunMetadata {
    pub executable: String,
    pub process_id: Option<u32>,
    pub argument_count: usize,
    pub arguments_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_provenance: Option<ManagedLiveProvenance>,
    pub startup_timeout_seconds: f64,
    pub readiness_probe_interval_ms: u128,
    pub shutdown_timeout_seconds: f64,
    pub startup_elapsed_seconds: f64,
    pub readiness_probe_attempts: u64,
    pub first_observed_source_frame_id: Option<String>,
    pub shutdown: Option<ManagedServerShutdown>,
}

/// Identity of live detector inputs and executable used by a managed run.
/// Only hashes and non-secret selection fields are written; camera URLs and
/// model/config file contents remain private to the machine running the audit.
#[derive(Debug, Serialize)]
pub struct ManagedLiveProvenance {
    pub effective_configuration_sha256: String,
    pub capture_settings_sha256: String,
    pub camera_profile_id: String,
    pub detector_model_sha256: String,
    pub model_manifest_sha256: String,
    pub source_checkpoint_sha256: String,
    pub detector_executable_sha256: String,
    pub capture_helper_executable_sha256: String,
    pub provider_selection: String,
    pub model_image_size: u64,
    pub ball_class_id: u64,
    pub cue_class_id: Option<u64>,
    pub confidence_threshold: f64,
    pub iou_threshold: f64,
}

#[derive(Debug, Serialize)]
pub struct ManagedServerShutdown {
    pub status: &'static str,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub forced: bool,
}

struct RunningManagedServer {
    child: tokio::process::Child,
    metadata: ManagedServerRunMetadata,
    started: Instant,
}

#[derive(Debug, Serialize)]
pub struct EffectiveRunConfiguration {
    pub warmup_seconds: f64,
    pub measured_seconds: f64,
    pub measurement_window_seconds: f64,
    pub repetitions: usize,
    pub soak: bool,
    pub soak_seconds: f64,
    pub connect_timeout_seconds: f64,
    pub application_heartbeat_interval_seconds: f64,
    pub summary_interval_seconds: f64,
    pub maximum_message_bytes: usize,
    pub maximum_samples_per_run: usize,
    pub subscriber_delay_ms: u128,
    pub resource_process_id: Option<u32>,
    pub resource_sample_interval_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub managed_live_process_resource_sampling: Option<EffectiveManagedLiveResourceConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_pipeline_timing_audit: Option<EffectiveLivePipelineTimingConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway_websocket_timing_audit: Option<EffectiveLivePipelineTimingConfiguration>,
}

#[derive(Debug, Serialize)]
pub struct EffectiveManagedLiveResourceConfiguration {
    pub process_role: &'static str,
    pub sample_interval_ms: u128,
    pub maximum_samples_per_window: usize,
    pub process_discovery: &'static str,
}

#[derive(Debug, Serialize)]
pub struct EffectiveLivePipelineTimingConfiguration {
    pub queue_capacity: usize,
    pub maximum_record_bytes: usize,
    pub maximum_retained_records: usize,
    pub maximum_retained_bytes: usize,
    pub connect_timeout_ms: u128,
    pub write_timeout_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct RunEnvironment {
    pub repository_revision: Option<String>,
    pub working_tree_dirty: Option<bool>,
    pub dirty_tree_status_sha256: Option<String>,
    pub operating_system: &'static str,
    pub architecture: &'static str,
    pub logical_cpu_count: Option<usize>,
    pub processor_brand: Option<String>,
    pub processor_frequency_mhz: Option<u64>,
    pub host_name: Option<String>,
    pub kernel_version: Option<String>,
    pub total_memory_bytes: Option<u64>,
    pub collector_version: &'static str,
}

#[derive(Debug, Serialize)]
pub struct InputMetadata {
    pub kind: &'static str,
    pub websocket_url: String,
    pub content_sha256: Option<String>,
    pub content_hash_status: &'static str,
    pub detector_model: Option<String>,
    pub detector_runtime: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WarmupSummary {
    pub requested_duration_seconds: f64,
    pub actual_duration_seconds: f64,
    pub state_messages: u64,
    pub unique_observed_source_frames: u64,
    pub termination_reason: &'static str,
    pub measurements_excluded: bool,
}

#[derive(Debug, Serialize)]
pub struct MeasuredRunReference {
    pub repetition: usize,
    pub report_path: String,
    pub details_path: String,
    pub report: StateReceiverReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_timing_summary_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_timing_details_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_timing_summary: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway_timing_summary_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway_timing_details_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gateway_timing_summary: Option<serde_json::Value>,
}

struct WindowTimingInput {
    repetition: usize,
    started_at_tick_ns: u64,
    ended_at_tick_ns: u64,
    sampled_process_id: Option<u32>,
    process_resource_status: Option<String>,
    details_path: PathBuf,
    state_trace_truncated: bool,
}

struct GatewayWindowTimingInput {
    repetition: usize,
    started_at_tick_ns: u64,
    ended_at_tick_ns: u64,
}

struct RunWindowsOutput {
    report: StateAuditRunReport,
    timing_inputs: Vec<WindowTimingInput>,
    gateway_timing_inputs: Vec<GatewayWindowTimingInput>,
}

struct WindowRunParameters<'a> {
    config: &'a StateAuditRunConfig,
    run_directory: &'a Path,
    run_id: &'a str,
    started_at_utc: String,
    environment: RunEnvironment,
    input: InputMetadata,
    live_audit_receiver: Option<&'a ManagedLiveAuditReceiver>,
    gateway_audit_receiver: Option<&'a ManagedLiveAuditReceiver>,
}

/// Run the configured warm-up and state-delivery measurement windows.
///
/// All filesystem writes happen after each receiver window has ended. The
/// receiver itself keeps bounded detail and percentile windows, and this
/// orchestration layer stores only one window's samples at a time.
pub async fn run_state_audit(config: &StateAuditRunConfig) -> Result<PathBuf> {
    config.validate()?;
    let run_directory = create_run_directory(&config.output_root)?;
    let run_id = run_directory
        .file_name()
        .and_then(|name| name.to_str())
        .context("generated audit directory has no valid run ID")?
        .to_owned();
    let started_at_utc = Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true);
    let environment = collect_environment();
    let mut input = InputMetadata {
        kind: "websocket_state_stream",
        websocket_url: config.url.clone(),
        content_sha256: None,
        content_hash_status: "unavailable_for_network_stream",
        detector_model: None,
        detector_runtime: None,
    };

    let mut child = None;
    let mut managed_metadata = None;
    let mut live_audit_receiver: Option<ManagedLiveAuditReceiver> = None;
    let mut gateway_audit_receiver: Option<ManagedLiveAuditReceiver> = None;
    let gateway_audit_run_id = config
        .gateway_websocket_timing_audit
        .as_ref()
        .map(|_| Uuid::new_v4().to_string());
    let result = async {
        if let Some(timing) = &config.live_pipeline_timing_audit {
            let receiver_config = ManagedLiveAuditReceiverConfig {
                address: SocketAddr::from(([127, 0, 0, 1], 0)),
                maximum_record_bytes: timing.maximum_record_bytes,
                queue_capacity: timing.queue_capacity,
                maximum_retained_records: timing.maximum_retained_records,
                maximum_retained_bytes: timing.maximum_retained_bytes,
                maximum_connections: 1,
            };
            live_audit_receiver = Some(
                ManagedLiveAuditReceiver::bind(receiver_config)
                    .await
                    .context("could not start the bounded live-timing receiver")?,
            );
        }
        if let Some(timing) = &config.gateway_websocket_timing_audit {
            let receiver_config = ManagedLiveAuditReceiverConfig {
                address: SocketAddr::from(([127, 0, 0, 1], 0)),
                maximum_record_bytes: timing.maximum_record_bytes,
                queue_capacity: timing.queue_capacity,
                maximum_retained_records: timing.maximum_retained_records,
                maximum_retained_bytes: timing.maximum_retained_bytes,
                maximum_connections: 1,
            };
            gateway_audit_receiver = Some(
                ManagedLiveAuditReceiver::bind(receiver_config)
                    .await
                    .context("could not start the bounded gateway-timing receiver")?,
            );
        }
        if let Some(launch) = &config.managed_server {
            let live_audit_arguments =
                match (&config.live_pipeline_timing_audit, &live_audit_receiver) {
                    (Some(timing), Some(receiver)) => Some((timing, receiver.listener_address())),
                    (None, None) => None,
                    _ => bail!("live timing receiver and producer configuration diverged"),
                };
            let gateway_audit_arguments = match (
                &config.gateway_websocket_timing_audit,
                &gateway_audit_receiver,
                gateway_audit_run_id.as_deref(),
            ) {
                (Some(timing), Some(receiver), Some(run_id)) => {
                    Some((timing, receiver.listener_address(), run_id))
                }
                (None, None, None) => None,
                _ => bail!("gateway timing receiver and producer configuration diverged"),
            };
            let running =
                spawn_managed_server(launch, live_audit_arguments, gateway_audit_arguments)?;
            let started = running.started;
            child = Some(running.child);
            managed_metadata = Some(running.metadata);
            if let Some(provenance) = managed_metadata
                .as_ref()
                .and_then(|metadata| metadata.live_provenance.as_ref())
            {
                input.detector_model = Some(format!("sha256:{}", provenance.detector_model_sha256));
                input.detector_runtime = Some(provenance.provider_selection.clone());
            }
            let managed_child = child
                .as_mut()
                .context("managed server child was not retained after launch")?;
            let readiness = wait_for_first_localization(config, launch, managed_child)
                .await
                .context("managed server did not publish a first observed state")?;
            if let Some(metadata) = &mut managed_metadata {
                metadata.startup_elapsed_seconds = started.elapsed().as_secs_f64();
                metadata.readiness_probe_attempts = readiness.probe_attempts;
                metadata.first_observed_source_frame_id = readiness.source_frame_id;
            }
        }

        let RunWindowsOutput {
            mut report,
            timing_inputs,
            gateway_timing_inputs,
        } = if let Some(managed_child) = &mut child {
            run_windows_while_server_alive(
                WindowRunParameters {
                    config,
                    run_directory: &run_directory,
                    run_id: &run_id,
                    started_at_utc: started_at_utc.clone(),
                    environment,
                    input,
                    live_audit_receiver: live_audit_receiver.as_ref(),
                    gateway_audit_receiver: gateway_audit_receiver.as_ref(),
                },
                managed_child,
            )
            .await?
        } else {
            run_windows(WindowRunParameters {
                config,
                run_directory: &run_directory,
                run_id: &run_id,
                started_at_utc: started_at_utc.clone(),
                environment,
                input,
                live_audit_receiver: None,
                gateway_audit_receiver: None,
            })
            .await?
        };

        if let (Some(managed_child), Some(metadata), Some(launch)) = (
            &mut child,
            &mut managed_metadata,
            config.managed_server.as_ref(),
        ) {
            metadata.shutdown =
                Some(stop_managed_server(managed_child, metadata.process_id, launch).await?);
        }
        if let Some(receiver) = live_audit_receiver.take() {
            let snapshot = receiver
                .shutdown()
                .await
                .context("could not drain the managed live-timing receiver")?;
            for timing_input in &timing_inputs {
                let run = report
                    .measured_runs
                    .get_mut(timing_input.repetition.saturating_sub(1))
                    .context("timing window did not have a matching measured run")?;
                let (summary, detail_records) = summarize_live_timing_window(
                    &snapshot,
                    timing_input,
                    config
                        .live_pipeline_timing_audit
                        .as_ref()
                        .map_or(config.receiver.maximum_samples, |timing| {
                            timing.maximum_retained_records
                        }),
                    &snapshot.measurement_clock_domain_id,
                    config.sample_managed_live_process_resources,
                )?;
                let (summary_path, details_path) = write_live_timing_artifacts(
                    &run_directory,
                    timing_input.repetition,
                    &summary,
                    &detail_records,
                )?;
                run.live_timing_summary_path = Some(summary_path);
                run.live_timing_details_path = Some(details_path);
                run.live_timing_summary = Some(summary);
            }
            report.live_pipeline_timing = Some(summarize_live_timing_receiver(&snapshot));
        }
        if let Some(receiver) = gateway_audit_receiver.take() {
            // Stop/reap the managed child before draining so the gateway's
            // independent exporter can emit its footer first.
            let snapshot = receiver
                .shutdown()
                .await
                .context("could not drain the managed gateway-timing receiver")?;
            let run_id = gateway_audit_run_id
                .as_deref()
                .context("gateway timing run ID is missing")?;
            for timing_input in &gateway_timing_inputs {
                let run = report
                    .measured_runs
                    .get_mut(timing_input.repetition.saturating_sub(1))
                    .context("gateway timing window did not have a matching measured run")?;
                let (summary, detail_records) = summarize_gateway_timing_window(
                    &snapshot,
                    timing_input,
                    config
                        .gateway_websocket_timing_audit
                        .as_ref()
                        .map_or(config.receiver.maximum_samples, |timing| {
                            timing.maximum_retained_records
                        }),
                    run_id,
                );
                let (summary_path, details_path) = write_gateway_timing_artifacts(
                    &run_directory,
                    timing_input.repetition,
                    &summary,
                    &detail_records,
                )?;
                run.gateway_timing_summary_path = Some(summary_path);
                run.gateway_timing_details_path = Some(details_path);
                run.gateway_timing_summary = Some(summary);
            }
            report.gateway_websocket_timing =
                Some(summarize_gateway_timing_receiver(&snapshot, run_id));
        }
        report.managed_server = managed_metadata.take();
        write_report_artifacts(&run_directory, &report)?;
        Ok::<PathBuf, anyhow::Error>(run_directory.clone())
    }
    .await;

    match result {
        Ok(path) => Ok(path),
        Err(error) => {
            if let (Some(managed_child), Some(metadata), Some(launch)) = (
                &mut child,
                &mut managed_metadata,
                config.managed_server.as_ref(),
            ) {
                if metadata.shutdown.is_none() {
                    metadata.shutdown =
                        stop_managed_server(managed_child, metadata.process_id, launch)
                            .await
                            .ok();
                }
            }
            if let Some(receiver) = live_audit_receiver.take() {
                if let Ok(snapshot) = receiver.shutdown().await {
                    let failure_path = run_directory.join("failure-live-audit.json");
                    if let Ok(file) = create_new_output(&failure_path, "live audit failure detail")
                    {
                        let mut writer = BufWriter::new(file);
                        let summary = summarize_live_timing_receiver(&snapshot);
                        let _ = serde_json::to_writer_pretty(&mut writer, &summary);
                        let _ = writer.write_all(b"\n");
                        let _ = writer.flush();
                    }
                }
            }
            if let Some(receiver) = gateway_audit_receiver.take() {
                if let Ok(snapshot) = receiver.shutdown().await {
                    let failure_path = run_directory.join("failure-gateway-audit.json");
                    if let Ok(file) =
                        create_new_output(&failure_path, "gateway audit failure detail")
                    {
                        let mut writer = BufWriter::new(file);
                        let summary = gateway_audit_run_id.as_deref().map_or_else(
                            || serde_json::json!({"schema_version": RUN_REPORT_SCHEMA_VERSION}),
                            |run_id| {
                                serde_json::to_value(summarize_gateway_timing_receiver(
                                    &snapshot, run_id,
                                ))
                                .unwrap_or_else(|_| serde_json::json!({}))
                            },
                        );
                        let _ = serde_json::to_writer_pretty(&mut writer, &summary);
                        let _ = writer.write_all(b"\n");
                        let _ = writer.flush();
                    }
                }
            }
            let failure_path = run_directory.join("failure.json");
            if let Ok(file) = create_new_output(&failure_path, "failure report") {
                let failure = serde_json::json!({
                    "schema_version": RUN_REPORT_SCHEMA_VERSION,
                    "tool": "billiards-performance-audit-run",
                    "run_id": run_id,
                    "error": format!("{error:#}"),
                    "failed_at_utc": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
                    "managed_server": managed_metadata,
                });
                let mut writer = BufWriter::new(file);
                let _ = serde_json::to_writer_pretty(&mut writer, &failure);
                let _ = writer.write_all(b"\n");
                let _ = writer.flush();
            }
            Err(error).with_context(|| {
                format!("state audit artifacts are in {}", run_directory.display())
            })
        }
    }
}

struct FirstLocalizationReadiness {
    probe_attempts: u64,
    source_frame_id: Option<String>,
}

fn spawn_managed_server(
    config: &ManagedServerConfig,
    live_audit: Option<(&LivePipelineTimingAuditConfig, SocketAddr)>,
    gateway_audit: Option<(&GatewayWebSocketTimingAuditConfig, SocketAddr, &str)>,
) -> Result<RunningManagedServer> {
    let live_provenance = managed_live_provenance(config)?;
    let mut effective_arguments = config.arguments.clone();
    if let Some((timing, address)) = live_audit {
        append_live_timing_server_arguments(&mut effective_arguments, timing, address);
    }
    if let Some((timing, address, run_id)) = gateway_audit {
        append_gateway_timing_server_arguments(&mut effective_arguments, timing, address, run_id);
    }
    let mut command = tokio::process::Command::new(&config.executable);
    command
        .args(&effective_arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Put the gateway and its supervised detector/capture descendants in
        // a private process group. If the runner itself is interrupted or the
        // gateway is force-stopped, the full live pipeline can still be
        // signalled as one bounded unit.
        command.as_std_mut().process_group(0);
    }
    if let Some(directory) = &config.working_directory {
        command.current_dir(directory);
    }
    let child = command.spawn().with_context(|| {
        format!(
            "could not start managed server {}",
            config.executable.display()
        )
    })?;
    let process_id = child.id();
    let argument_count = effective_arguments.len();
    let mut argument_hasher = Sha256::new();
    for argument in &effective_arguments {
        argument_hasher.update((argument.len() as u64).to_be_bytes());
        argument_hasher.update(argument.as_bytes());
    }
    Ok(RunningManagedServer {
        child,
        metadata: ManagedServerRunMetadata {
            executable: config.executable.display().to_string(),
            process_id,
            argument_count,
            arguments_sha256: format!("{:x}", argument_hasher.finalize()),
            live_provenance,
            startup_timeout_seconds: config.startup_timeout.as_secs_f64(),
            readiness_probe_interval_ms: config.readiness_probe_interval.as_millis(),
            shutdown_timeout_seconds: config.shutdown_timeout.as_secs_f64(),
            startup_elapsed_seconds: 0.0,
            readiness_probe_attempts: 0,
            first_observed_source_frame_id: None,
            shutdown: None,
        },
        started: Instant::now(),
    })
}

fn append_live_timing_server_arguments(
    arguments: &mut Vec<String>,
    timing: &LivePipelineTimingAuditConfig,
    address: SocketAddr,
) {
    for (flag, value) in [
        ("--live-audit-tcp", address.to_string()),
        (
            "--live-audit-queue-capacity",
            timing.queue_capacity.to_string(),
        ),
        (
            "--live-audit-max-record-bytes",
            timing.maximum_record_bytes.to_string(),
        ),
        (
            "--live-audit-connect-timeout-ms",
            timing.connect_timeout.as_millis().to_string(),
        ),
        (
            "--live-audit-write-timeout-ms",
            timing.write_timeout.as_millis().to_string(),
        ),
    ] {
        arguments.push(flag.to_owned());
        arguments.push(value);
    }
}

fn append_gateway_timing_server_arguments(
    arguments: &mut Vec<String>,
    timing: &GatewayWebSocketTimingAuditConfig,
    address: SocketAddr,
    run_id: &str,
) {
    for (flag, value) in [
        ("--gateway-audit-tcp", address.to_string()),
        (
            "--gateway-audit-queue-capacity",
            timing.queue_capacity.to_string(),
        ),
        (
            "--gateway-audit-max-record-bytes",
            timing.maximum_record_bytes.to_string(),
        ),
        (
            "--gateway-audit-connect-timeout-ms",
            timing.connect_timeout.as_millis().to_string(),
        ),
        (
            "--gateway-audit-write-timeout-ms",
            timing.write_timeout.as_millis().to_string(),
        ),
        ("--gateway-audit-run-id", run_id.to_owned()),
    ] {
        arguments.push(flag.to_owned());
        arguments.push(value);
    }
}

fn managed_live_provenance(server: &ManagedServerConfig) -> Result<Option<ManagedLiveProvenance>> {
    if !server
        .arguments
        .iter()
        .any(|argument| argument == "--start-live")
    {
        return Ok(None);
    }

    use crate::capture_settings_wire::build_capture_settings_v3;
    use crate::configuration_authority::{AuthorityLimits, ConfigurationAuthority};
    use crate::live_supervisor::resolve_live_executables;
    use serde_json::Value;

    let working_directory = match &server.working_directory {
        Some(path) => path.clone(),
        None => std::env::current_dir().context("could not resolve current directory")?,
    };
    let project_root = unique_argument_value(&server.arguments, "--project-root")?
        .map(PathBuf::from)
        .unwrap_or_else(|| working_directory.clone());
    if !project_root.is_absolute() {
        bail!("managed live project root must be absolute");
    }

    let raw_configuration_path = unique_argument_value(&server.arguments, "--config")?
        .context("managed live server requires an explicit --config argument")?;
    let configuration_path = resolve_from_directory(raw_configuration_path, &working_directory);
    let authority = ConfigurationAuthority::new(configuration_path, AuthorityLimits::default())
        .map_err(anyhow::Error::msg)
        .context("could not initialize the managed live configuration authority")?;
    let configuration = authority
        .effective_configuration()
        .map_err(anyhow::Error::msg)
        .context("could not read the managed live effective configuration")?;
    let live = configuration
        .pointer("/perception/rust_live")
        .and_then(Value::as_object)
        .context("effective configuration has no perception.rust_live object")?;
    let capture_settings = build_capture_settings_v3(&configuration)
        .context("could not resolve managed live capture settings")?;
    let camera_profile_id = capture_settings
        .pointer("/requested/camera_profile_id")
        .and_then(Value::as_str)
        .context("resolved capture settings have no camera profile identity")?
        .to_owned();

    let effective_configuration_bytes = serde_json::to_vec(&configuration)
        .context("could not serialize effective managed live configuration")?;
    let capture_settings_bytes = serde_json::to_vec(&capture_settings)
        .context("could not serialize managed live capture settings")?;
    let model_path = configured_project_path(live, "onnx_path", &project_root)?;
    let model_manifest_path = PathBuf::from(format!("{}.manifest.json", model_path.display()));
    let checkpoint_path = configured_project_path(live, "source_checkpoint_path", &project_root)?;
    let server_executable = resolve_program_path(&server.executable, &working_directory)?
        .canonicalize()
        .context("could not canonicalize managed server executable")?;
    let sibling_directory = server_executable
        .parent()
        .context("managed server executable has no parent directory")?;
    let executables = resolve_live_executables(
        &configuration,
        &project_root,
        unique_argument_value(&server.arguments, "--live-executable")?.map(Path::new),
        unique_argument_value(&server.arguments, "--capture-helper-executable")?.map(Path::new),
        Some(sibling_directory),
    )
    .map_err(anyhow::Error::msg)
    .context("could not resolve the managed live detector executables")?;
    let detector_executable = executables
        .detector
        .canonicalize()
        .context("could not canonicalize managed live detector executable")?;
    let capture_helper_executable = executables
        .capture_helper
        .canonicalize()
        .context("could not canonicalize managed live capture helper executable")?;
    let provider_selection = required_configuration_string(live, "provider")?;
    let model_image_size = live
        .get("image_size")
        .and_then(Value::as_u64)
        .context("perception.rust_live.image_size must be a positive integer")?;
    let ball_class_id = live
        .get("ball_class_id")
        .and_then(Value::as_u64)
        .context("perception.rust_live.ball_class_id must be a non-negative integer")?;
    let cue_class_id = live.get("cue_class_id").and_then(Value::as_u64);
    let confidence_threshold = live
        .get("confidence")
        .and_then(Value::as_f64)
        .context("perception.rust_live.confidence must be a finite number")?;
    let iou_threshold = live
        .get("iou")
        .and_then(Value::as_f64)
        .context("perception.rust_live.iou must be a finite number")?;

    Ok(Some(ManagedLiveProvenance {
        effective_configuration_sha256: sha256_bytes(&effective_configuration_bytes),
        capture_settings_sha256: sha256_bytes(&capture_settings_bytes),
        camera_profile_id,
        detector_model_sha256: sha256_file(&model_path)?,
        model_manifest_sha256: sha256_file(&model_manifest_path)?,
        source_checkpoint_sha256: sha256_file(&checkpoint_path)?,
        detector_executable_sha256: sha256_file(&detector_executable)?,
        capture_helper_executable_sha256: sha256_file(&capture_helper_executable)?,
        provider_selection,
        model_image_size,
        ball_class_id,
        cue_class_id,
        confidence_threshold,
        iou_threshold,
    }))
}

fn unique_argument_value<'a>(arguments: &'a [String], name: &str) -> Result<Option<&'a str>> {
    let mut found = None;
    for (index, argument) in arguments.iter().enumerate() {
        let value = if let Some(value) = argument.strip_prefix(&format!("{name}=")) {
            Some(value)
        } else if argument == name {
            Some(
                arguments
                    .get(index + 1)
                    .context("managed server option is missing its value")?
                    .as_str(),
            )
        } else {
            None
        };
        if let Some(value) = value {
            if found.replace(value).is_some() {
                bail!("managed server has more than one {name} argument");
            }
        }
    }
    Ok(found)
}

fn resolve_from_directory(value: &str, directory: &Path) -> PathBuf {
    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        directory.join(path)
    }
}

fn configured_project_path(
    live: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    project_root: &Path,
) -> Result<PathBuf> {
    let configured = required_configuration_string(live, key)?;
    let path = Path::new(&configured);
    Ok(if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_root.join(path)
    })
}

fn required_configuration_string(
    values: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<String> {
    values
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .with_context(|| format!("perception.rust_live.{key} must be a non-empty string"))
}

fn resolve_program_path(program: &Path, working_directory: &Path) -> Result<PathBuf> {
    if program.is_absolute() {
        return Ok(program.to_path_buf());
    }
    if program.components().count() > 1 {
        return Ok(working_directory.join(program));
    }
    for directory in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let directory = if directory.is_absolute() {
            directory
        } else {
            working_directory.join(directory)
        };
        let candidate = directory.join(program);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!(
        "could not resolve managed server executable {}",
        program.display()
    )
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("could not open provenance input {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; PROVENANCE_HASH_BUFFER_SIZE];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("could not read provenance input {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

async fn wait_for_first_localization(
    run: &StateAuditRunConfig,
    launch: &ManagedServerConfig,
    child: &mut tokio::process::Child,
) -> Result<FirstLocalizationReadiness> {
    let started = Instant::now();
    let mut attempts = 0u64;
    let mut last_connection_error = None;
    while started.elapsed() < launch.startup_timeout {
        if let Some(status) = child
            .try_wait()
            .context("could not check managed server process state")?
        {
            bail!("managed server exited before readiness: {status}");
        }
        let remaining = launch.startup_timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            break;
        }
        let probe_duration = launch.readiness_probe_interval.min(remaining);
        let mut probe = run.receiver.clone();
        probe.duration = probe_duration;
        probe.connect_timeout = probe.connect_timeout.min(probe_duration);
        probe.maximum_samples = 1;
        probe.resource_sample_process_id = None;
        attempts = attempts.saturating_add(1);
        match connect_and_wait_for_first_fresh_observed_localization(&run.url, &probe).await {
            Ok(output) => {
                if output.report.termination_reason == "first_fresh_observed_localization" {
                    let sample = output.retained_samples.last().context(
                        "readiness receiver stopped without retaining its qualifying state",
                    )?;
                    return Ok(FirstLocalizationReadiness {
                        probe_attempts: attempts,
                        source_frame_id: sample.source_frame_id.clone(),
                    });
                }
            }
            Err(error) => last_connection_error = Some(format!("{error:#}")),
        }
        if let Some(status) = child
            .try_wait()
            .context("could not check managed server process state")?
        {
            bail!("managed server exited before readiness: {status}");
        }
        let remaining = launch.startup_timeout.saturating_sub(started.elapsed());
        if !remaining.is_zero() {
            tokio::time::sleep(launch.readiness_probe_interval.min(remaining)).await;
        }
    }
    let connection_detail = last_connection_error
        .map(|error| format!("; last WebSocket error: {error}"))
        .unwrap_or_default();
    bail!(
        "no fresh observed source state arrived within {:.2}s after {attempts} readiness probes{connection_detail}",
        launch.startup_timeout.as_secs_f64()
    )
}

async fn run_windows_while_server_alive(
    parameters: WindowRunParameters<'_>,
    child: &mut tokio::process::Child,
) -> Result<RunWindowsOutput> {
    tokio::select! {
        result = run_windows(parameters) => result,
        status = child.wait() => {
            let status = status.context("could not wait for managed server process")?;
            bail!("managed server exited during the audit windows: {status}");
        }
    }
}

async fn stop_managed_server(
    child: &mut tokio::process::Child,
    process_group_id: Option<u32>,
    config: &ManagedServerConfig,
) -> Result<ManagedServerShutdown> {
    let parent_status_before_shutdown = child
        .try_wait()
        .context("could not check managed server before shutdown")?;
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        let process_group_id =
            process_group_id.context("managed server process group has no operating-system ID")?;
        let process_group_id = libc::pid_t::try_from(process_group_id)
            .context("managed server process group ID is outside the platform range")?;
        let group_existed_before_shutdown = process_group_exists(process_group_id)?;
        if !group_existed_before_shutdown {
            let status = parent_status_before_shutdown
                .context("managed server process group exited before its process was reaped")?;
            return Ok(shutdown_from_status("already_exited", status, false));
        }

        signal_process_group(process_group_id, libc::SIGTERM)
            .context("could not signal managed server process group to stop")?;
        let deadline = Instant::now() + config.shutdown_timeout;
        let mut parent_status = parent_status_before_shutdown;
        loop {
            if parent_status.is_none() {
                parent_status = child
                    .try_wait()
                    .context("could not check managed server while shutting down")?;
            }
            if !process_group_exists(process_group_id)? {
                if let Some(status) = parent_status {
                    return Ok(shutdown_from_status(
                        if status.success() || status.signal() == Some(libc::SIGTERM) {
                            "terminated_gracefully"
                        } else {
                            "exited_during_shutdown"
                        },
                        status,
                        false,
                    ));
                }
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            tokio::time::sleep(remaining.min(Duration::from_millis(20))).await;
        }

        signal_process_group(process_group_id, libc::SIGKILL)
            .context("could not force-stop the managed server process group")?;
        let status = if let Some(status) = parent_status {
            status
        } else {
            tokio::time::timeout(config.shutdown_timeout, child.wait())
                .await
                .context("managed server did not exit after forced termination")?
                .context("could not reap force-stopped managed server")?
        };
        let forced_deadline = Instant::now() + config.shutdown_timeout;
        while process_group_exists(process_group_id)? {
            let remaining = forced_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                bail!("managed server descendants remain after process-group termination");
            }
            tokio::time::sleep(remaining.min(Duration::from_millis(20))).await;
        }
        Ok(shutdown_from_status("force_killed", status, true))
    }
    #[cfg(not(unix))]
    {
        if let Some(status) = parent_status_before_shutdown {
            return Ok(shutdown_from_status("already_exited", status, false));
        }
        child
            .start_kill()
            .context("could not stop the managed server")?;
        let status = tokio::time::timeout(config.shutdown_timeout, child.wait())
            .await
            .context("managed server did not exit after termination")?
            .context("could not reap managed server")?;
        Ok(shutdown_from_status("terminated", status, false))
    }
}

#[cfg(unix)]
fn signal_process_group(process_group_id: libc::pid_t, signal: libc::c_int) -> Result<()> {
    // SAFETY: process_group_id was obtained from this runner's child and the
    // child was started in a private group, so the negative ID cannot target
    // the runner or another application process.
    let result = unsafe { libc::kill(-process_group_id, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(error).context("could not signal managed server process group")
}

#[cfg(unix)]
fn process_group_exists(process_group_id: libc::pid_t) -> Result<bool> {
    // SAFETY: signal zero only probes for the private group created for this
    // runner-owned child and does not deliver a signal to its members.
    let result = unsafe { libc::kill(-process_group_id, 0) };
    if result == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(false);
    }
    if error.raw_os_error() == Some(libc::EPERM) {
        return Ok(true);
    }
    Err(error).context("could not inspect managed server process group")
}

fn shutdown_from_status(
    status_name: &'static str,
    status: ExitStatus,
    forced: bool,
) -> ManagedServerShutdown {
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt;
        status.signal()
    };
    #[cfg(not(unix))]
    let signal = None;
    ManagedServerShutdown {
        status: status_name,
        exit_code: status.code(),
        signal,
        forced,
    }
}

fn validate_loopback_websocket_url(url: &str) -> Result<()> {
    let authority = url
        .strip_prefix("ws://")
        .and_then(|rest| rest.split('/').next())
        .unwrap_or_default();
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        bracketed.split(']').next().unwrap_or_default()
    } else {
        authority
            .rsplit_once(':')
            .map_or(authority, |(host, _)| host)
    };
    let loopback = host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback());
    if !loopback {
        bail!("managed server WebSocket URL must use a loopback address");
    }
    Ok(())
}

async fn run_windows(parameters: WindowRunParameters<'_>) -> Result<RunWindowsOutput> {
    let WindowRunParameters {
        config,
        run_directory,
        run_id,
        started_at_utc,
        environment,
        input,
        live_audit_receiver,
        gateway_audit_receiver,
    } = parameters;
    let warmup = if config.warmup.is_zero() {
        None
    } else {
        eprintln!("audit warm-up: {:.2}s", config.warmup.as_secs_f64());
        let mut warmup_receiver = config.receiver.clone();
        warmup_receiver.duration = config.warmup;
        warmup_receiver.maximum_samples = 1;
        warmup_receiver.resource_sample_process_id = None;
        let output = connect_and_collect(&config.url, &warmup_receiver)
            .await
            .context("state audit warm-up failed")?;
        if config.managed_server.is_some()
            && output.report.freshness.unique_observed_source_observations == 0
        {
            bail!("managed server produced no fresh observed state during warm-up");
        }
        Some(WarmupSummary {
            requested_duration_seconds: config.warmup.as_secs_f64(),
            actual_duration_seconds: output.report.elapsed_seconds,
            state_messages: output.report.counts.state_messages,
            unique_observed_source_frames: output
                .report
                .freshness
                .unique_observed_source_observations,
            termination_reason: output.report.termination_reason,
            measurements_excluded: true,
        })
    };

    let mut measured_runs = Vec::with_capacity(config.measurement_count());
    let mut timing_inputs = Vec::with_capacity(config.measurement_count());
    let mut gateway_timing_inputs = Vec::with_capacity(config.measurement_count());
    for repetition in 1..=config.measurement_count() {
        let measured_duration = config.measurement_duration();
        eprintln!(
            "audit measured run {repetition}/{}: {:.2}s{}",
            config.measurement_count(),
            measured_duration.as_secs_f64(),
            if config.soak { " (soak)" } else { "" },
        );
        let mut receiver = config.receiver.clone();
        receiver.duration = measured_duration;
        let selected_managed_process_id = if config.sample_managed_live_process_resources {
            let audit = live_audit_receiver
                .context("managed live process sampling has no timing receiver")?;
            match audit.latest_producer_process_id() {
                Some(process_id) => Some(process_id),
                None => audit
                    .wait_for_producer_process_id(config.receiver.connect_timeout)
                    .await
                    .context("could not wait for the managed detector process ID")?,
            }
        } else {
            None
        };
        if config.sample_managed_live_process_resources {
            receiver.resource_sample_process_id = selected_managed_process_id;
        }
        let output = connect_and_collect(&config.url, &receiver)
            .await
            .with_context(|| format!("state audit measured run {repetition} failed"))?;
        let timing_started_at_tick_ns = live_audit_receiver
            .zip(output.measurement_started_at)
            .map(|(audit, started)| audit.measurement_clock_tick_at(started));
        let timing_ended_at_tick_ns = live_audit_receiver
            .zip(output.measurement_ended_at)
            .map(|(audit, ended)| audit.measurement_clock_tick_at(ended));
        let gateway_timing_started_at_tick_ns = gateway_audit_receiver
            .zip(output.measurement_started_at)
            .map(|(audit, started)| audit.measurement_clock_tick_at(started));
        let gateway_timing_ended_at_tick_ns = gateway_audit_receiver
            .zip(output.measurement_ended_at)
            .map(|(audit, ended)| audit.measurement_clock_tick_at(ended));
        if config.managed_server.is_some()
            && output.report.freshness.unique_observed_source_observations == 0
        {
            bail!("managed server produced no fresh observed state in measured run {repetition}");
        }
        let (report_path, details_path) =
            write_window_artifacts(run_directory, repetition, &output)?;
        let sampled_process_id = output
            .report
            .process_resources
            .as_ref()
            .map(|resources| resources.process_id);
        let process_resource_status = output
            .report
            .process_resources
            .as_ref()
            .map(|resources| resources.status.to_owned());
        measured_runs.push(MeasuredRunReference {
            repetition,
            report_path,
            details_path,
            report: output.report,
            live_timing_summary_path: None,
            live_timing_details_path: None,
            live_timing_summary: None,
            gateway_timing_summary_path: None,
            gateway_timing_details_path: None,
            gateway_timing_summary: None,
        });
        if let (Some(started_at_tick_ns), Some(ended_at_tick_ns)) =
            (timing_started_at_tick_ns, timing_ended_at_tick_ns)
        {
            timing_inputs.push(WindowTimingInput {
                repetition,
                started_at_tick_ns,
                ended_at_tick_ns,
                sampled_process_id,
                process_resource_status,
                details_path: run_directory
                    .join("runs")
                    .join(format!("run-{repetition:03}.jsonl")),
                state_trace_truncated: measured_runs
                    .last()
                    .is_some_and(|run| run.report.detail_retention.state_samples_truncated),
            });
        }
        if let (Some(started_at_tick_ns), Some(ended_at_tick_ns)) = (
            gateway_timing_started_at_tick_ns,
            gateway_timing_ended_at_tick_ns,
        ) {
            gateway_timing_inputs.push(GatewayWindowTimingInput {
                repetition,
                started_at_tick_ns,
                ended_at_tick_ns,
            });
        }
    }

    let configuration = EffectiveRunConfiguration {
        warmup_seconds: config.warmup.as_secs_f64(),
        measured_seconds: config.measured.as_secs_f64(),
        measurement_window_seconds: config.measurement_duration().as_secs_f64(),
        repetitions: config.measurement_count(),
        soak: config.soak,
        soak_seconds: config.soak_duration.as_secs_f64(),
        connect_timeout_seconds: config.receiver.connect_timeout.as_secs_f64(),
        application_heartbeat_interval_seconds: config
            .receiver
            .application_heartbeat_interval
            .as_secs_f64(),
        summary_interval_seconds: config.receiver.summary_interval.as_secs_f64(),
        maximum_message_bytes: config.receiver.maximum_message_bytes,
        maximum_samples_per_run: config.receiver.maximum_samples,
        subscriber_delay_ms: config.receiver.subscriber_delay.as_millis(),
        resource_process_id: config.receiver.resource_sample_process_id,
        resource_sample_interval_ms: config.receiver.resource_sample_interval.as_millis(),
        managed_live_process_resource_sampling: config
            .sample_managed_live_process_resources
            .then_some(EffectiveManagedLiveResourceConfiguration {
                process_role: "rust_live_inference_child",
                sample_interval_ms: config.receiver.resource_sample_interval.as_millis(),
                maximum_samples_per_window: config.receiver.maximum_samples,
                process_discovery: "latest_positive_pid_from_live_timing_records",
            }),
        live_pipeline_timing_audit: config.live_pipeline_timing_audit.as_ref().map(|timing| {
            EffectiveLivePipelineTimingConfiguration {
                queue_capacity: timing.queue_capacity,
                maximum_record_bytes: timing.maximum_record_bytes,
                maximum_retained_records: timing.maximum_retained_records,
                maximum_retained_bytes: timing.maximum_retained_bytes,
                connect_timeout_ms: timing.connect_timeout.as_millis(),
                write_timeout_ms: timing.write_timeout.as_millis(),
            }
        }),
        gateway_websocket_timing_audit: config.gateway_websocket_timing_audit.as_ref().map(
            |timing| EffectiveLivePipelineTimingConfiguration {
                queue_capacity: timing.queue_capacity,
                maximum_record_bytes: timing.maximum_record_bytes,
                maximum_retained_records: timing.maximum_retained_records,
                maximum_retained_bytes: timing.maximum_retained_bytes,
                connect_timeout_ms: timing.connect_timeout.as_millis(),
                write_timeout_ms: timing.write_timeout.as_millis(),
            },
        ),
    };
    let mut limitations = vec![
        "This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.",
        "Detector accuracy is unavailable without temporally aligned ground-truth annotations.",
        "A network state stream has no input-video content hash.",
        "State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.",
    ];
    if input.detector_model.is_none() {
        limitations.push(
            "Detector model and configured runtime identity are unavailable for this externally managed state stream.",
        );
    } else {
        limitations.push(
            "The recorded provider is the configured selection; actual per-node accelerator placement and dynamic runtime version are unavailable.",
        );
    }
    if config.managed_server.is_some() {
        if config.sample_managed_live_process_resources {
            limitations.push("Managed live process sampling follows the latest detector PID in timing records for each measured window; detector restarts during a window are reported as incomplete process-resource evidence.");
        } else if config.receiver.resource_sample_process_id.is_none() {
            limitations.push("Managed-server mode qualifies state delivery and process lifecycle; managed detector CPU/RSS sampling is disabled unless --sample-managed-live-process-resources is selected.");
        }
    }
    if config.live_pipeline_timing_audit.is_some() {
        limitations.push(
            "Live-pipeline timing is diagnostic and opt-in; it does not provide temporal ground-truth accuracy evidence or camera-exposure timing.",
        );
    }
    if config.gateway_websocket_timing_audit.is_some() {
        limitations.push(
            "Gateway timing reports publication-to-selection and application socket-send-call durations; actual client receipt and physical display latency are unavailable.",
        );
        limitations.push(
            "Gateway timing uses an independent receiver clock domain and does not compare detector producer ticks or drive detector resource sampling.",
        );
    }
    Ok(RunWindowsOutput {
        report: StateAuditRunReport {
            schema_version: RUN_REPORT_SCHEMA_VERSION,
            tool: "billiards-performance-audit-run",
            run_id: run_id.to_owned(),
            mode: if config.soak { "soak" } else { "repeated" },
            started_at_utc,
            finished_at_utc: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
            configuration,
            environment,
            input,
            warmup,
            measured_runs,
            live_pipeline_timing: None,
            gateway_websocket_timing: None,
            managed_server: None,
            limitations,
        },
        timing_inputs,
        gateway_timing_inputs,
    })
}

fn create_run_directory(output_root: &Path) -> Result<PathBuf> {
    fs::create_dir_all(output_root).with_context(|| {
        format!(
            "could not create audit output root {}",
            output_root.display()
        )
    })?;
    let stamp = Utc::now().format("%Y%m%d-%H%M%S");
    for _ in 0..16 {
        let suffix = &Uuid::new_v4().simple().to_string()[..8];
        let path = output_root.join(format!("{stamp}-{suffix}"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("could not create audit run {}", path.display()));
            }
        }
    }
    bail!("could not allocate a unique audit run directory")
}

fn summarize_live_timing_window(
    snapshot: &ManagedLiveAuditSnapshot,
    input: &WindowTimingInput,
    maximum_samples: usize,
    clock_domain_id: &str,
    sample_managed_process: bool,
) -> Result<(serde_json::Value, Vec<serde_json::Value>)> {
    let mut accumulator = RustLiveTimingAccumulator::new(maximum_samples)?;
    accumulator.begin_measurement(input.started_at_tick_ns);
    let mut detail_records = Vec::new();
    let mut observed_process_ids = std::collections::BTreeSet::new();
    for received in &snapshot.retained_records {
        if received.received_at_tick_ns < input.started_at_tick_ns
            || received.received_at_tick_ns > input.ended_at_tick_ns
        {
            continue;
        }
        if let Some(process_id) = received
            .record
            .get("process_id")
            .and_then(serde_json::Value::as_u64)
            .and_then(|process_id| u32::try_from(process_id).ok())
            .filter(|process_id| *process_id > 0)
        {
            observed_process_ids.insert(process_id);
        }
        let mut record = received.record.clone();
        if let Some(object) = record.as_object_mut() {
            object.insert(
                "audit_receiver_monotonic_ns".to_owned(),
                serde_json::json!(received.received_at_tick_ns),
            );
        }
        accumulator.add(record.clone());
        detail_records.push(record);
    }
    // The producer footer is emitted after all measurements, when the managed
    // process shuts down. Include it for completeness status without moving it
    // into the measured window or joining it to a state frame.
    if let Some(footer) = &snapshot.producer_footer {
        let mut footer = footer.clone();
        if let Some(object) = footer.as_object_mut() {
            object.insert(
                "audit_receiver_monotonic_ns".to_owned(),
                serde_json::json!(snapshot.measurement_clock_tick_ns),
            );
        }
        accumulator.add(footer.clone());
        detail_records.push(footer);
    }
    accumulator.end_measurement(input.ended_at_tick_ns);

    let state_trace = timing_state_trace(&input.details_path, maximum_samples)?;
    let mut summary = accumulator.summary(&state_trace, input.state_trace_truncated);
    if let Some(window) = summary
        .get_mut("measurement_monotonic_window_ns")
        .and_then(serde_json::Value::as_object_mut)
    {
        window.insert(
            "clock".to_owned(),
            serde_json::json!("managed_live_audit_receiver_monotonic"),
        );
        window.insert(
            "clock_domain_id".to_owned(),
            serde_json::json!(clock_domain_id),
        );
    }
    if let Some(object) = summary.as_object_mut() {
        object.insert(
            "receiver_loss_counts_run_total".to_owned(),
            serde_json::to_value(&snapshot.losses)
                .context("could not serialize live timing receiver loss counts")?,
        );
        let receiver = summarize_live_timing_receiver(snapshot);
        let trace_truncated = object
            .get("state_timing_correlation")
            .and_then(|value| value.get("timing_trace_truncated"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if sample_managed_process {
            let unsampled_process_ids = observed_process_ids
                .iter()
                .copied()
                .filter(|process_id| Some(*process_id) != input.sampled_process_id)
                .collect::<Vec<_>>();
            let restart_during_window = observed_process_ids.len() > 1;
            let resource_sampling_status = if input.sampled_process_id.is_none()
                || input.process_resource_status.as_deref() != Some("available")
                || !unsampled_process_ids.is_empty()
                || observed_process_ids.is_empty()
                || trace_truncated
                || receiver.status != "complete"
            {
                if input.sampled_process_id.is_none() || observed_process_ids.is_empty() {
                    "unavailable"
                } else {
                    "partial"
                }
            } else {
                "complete"
            };
            object.insert(
                "managed_process_resource_sampling".to_owned(),
                serde_json::json!({
                    "status": resource_sampling_status,
                    "sampled_process_id": input.sampled_process_id,
                    "sampled_process_status": input.process_resource_status.clone(),
                    "observed_process_ids": observed_process_ids,
                    "unsampled_process_ids": unsampled_process_ids,
                    "restart_during_window": restart_during_window,
                }),
            );
        }
        if receiver.status != "complete" || trace_truncated {
            object.insert("status".to_owned(), serde_json::json!("partial"));
        }
    }
    Ok((summary, detail_records))
}

fn timing_state_trace(path: &Path, maximum_samples: usize) -> Result<Vec<serde_json::Value>> {
    let file = File::open(path)
        .with_context(|| format!("could not open state detail trace {}", path.display()))?;
    let mut trace = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.with_context(|| format!("could not read state detail trace {}", path.display()))?;
        let sample: serde_json::Value = serde_json::from_str(&line).with_context(|| {
            format!(
                "invalid state detail JSON at {}:{}",
                path.display(),
                line_index + 1
            )
        })?;
        let Some(identity) = sample
            .get("observation_identity")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        if sample
            .get("ball_positions_status")
            .and_then(serde_json::Value::as_str)
            != Some("observed")
        {
            continue;
        }
        let mut fields = identity.split('\u{1f}');
        let (Some(engine), Some(session), Some(frame), None) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let Some(sequence) = sample
            .get("transport_sequence")
            .and_then(serde_json::Value::as_u64)
        else {
            continue;
        };
        trace.push(serde_json::json!({
            "engine_instance_id": engine,
            "source_session_id": session,
            "source_frame_id": frame,
            "source_transport_sequence": sequence,
        }));
        if trace.len() >= maximum_samples {
            break;
        }
    }
    Ok(trace)
}

fn summarize_live_timing_receiver(
    snapshot: &ManagedLiveAuditSnapshot,
) -> ManagedLivePipelineTimingSummary {
    let footer_complete = snapshot.producer_footer.as_ref().is_some_and(|footer| {
        footer
            .get("export_status")
            .and_then(serde_json::Value::as_str)
            == Some("complete")
            && footer
                .get("dropped_records")
                .and_then(serde_json::Value::as_u64)
                == Some(0)
    });
    let receiver_losses = &snapshot.losses;
    let receiver_loss_free = receiver_losses.malformed_records == 0
        && receiver_losses.oversized_records == 0
        && receiver_losses.queue_dropped_records == 0
        && receiver_losses.retention_evictions == 0
        && receiver_losses.retention_oversize_drops == 0
        && receiver_losses.connection_limit_drops == 0;
    ManagedLivePipelineTimingSummary {
        status: if footer_complete && receiver_loss_free {
            "complete"
        } else {
            "partial"
        },
        listener_address: snapshot.listener_address.to_string(),
        measurement_clock_domain_id: snapshot.measurement_clock_domain_id.clone(),
        stream_counts: snapshot.stream_counts.clone(),
        producer_footer: snapshot.producer_footer.clone(),
        losses: receiver_losses.clone(),
        retained_record_count: snapshot.retained_records.len(),
        retained_record_bytes: snapshot.retained_record_bytes,
        maximum_retained_records: snapshot.maximum_retained_records,
        maximum_retained_bytes: snapshot.maximum_retained_bytes,
    }
}

fn summarize_gateway_timing_receiver(
    snapshot: &ManagedLiveAuditSnapshot,
    run_id: &str,
) -> ManagedGatewayWebSocketTimingSummary {
    let footer_complete = snapshot.producer_footer.as_ref().is_some_and(|footer| {
        footer
            .get("export_status")
            .and_then(serde_json::Value::as_str)
            == Some("complete")
            && footer
                .get("dropped_records")
                .and_then(serde_json::Value::as_u64)
                == Some(0)
    });
    let losses = &snapshot.losses;
    let receiver_loss_free = losses.malformed_records == 0
        && losses.oversized_records == 0
        && losses.queue_dropped_records == 0
        && losses.retention_evictions == 0
        && losses.retention_oversize_drops == 0
        && losses.connection_limit_drops == 0
        && losses.connection_task_failures == 0
        && losses.transport_errors == 0;
    ManagedGatewayWebSocketTimingSummary {
        status: if footer_complete && receiver_loss_free {
            "complete"
        } else {
            "partial"
        },
        run_id: run_id.to_owned(),
        listener_address: snapshot.listener_address.to_string(),
        measurement_clock_domain_id: snapshot.measurement_clock_domain_id.clone(),
        stream_counts: snapshot.stream_counts.clone(),
        producer_footer: snapshot.producer_footer.clone(),
        footer_complete,
        losses: losses.clone(),
        retained_record_count: snapshot.retained_records.len(),
        retained_record_bytes: snapshot.retained_record_bytes,
        maximum_retained_records: snapshot.maximum_retained_records,
        maximum_retained_bytes: snapshot.maximum_retained_bytes,
        producer_process_id: snapshot.latest_producer_process_id,
    }
}

fn summarize_gateway_timing_window(
    snapshot: &ManagedLiveAuditSnapshot,
    input: &GatewayWindowTimingInput,
    maximum_samples: usize,
    run_id: &str,
) -> (serde_json::Value, Vec<serde_json::Value>) {
    use std::collections::{BTreeMap, BTreeSet};

    let mut detail_records = Vec::new();
    let mut outcomes = BTreeMap::<String, u64>::new();
    let mut gateway_send_records = 0_u64;
    let mut unique_publications = BTreeSet::<(u64, u64)>::new();
    let mut client_ids = BTreeSet::<u64>::new();
    let mut repeated_publications = 0_u64;
    let mut repeated_source_observations = 0_u64;
    let mut publication_sequence_gaps = 0_u64;
    let mut source_sequence_gaps = 0_u64;
    let mut payload_bytes = Vec::new();
    let mut publication_to_selection = Vec::new();
    let mut watch_wakeup_to_consume = Vec::new();
    let mut message_assembly = Vec::new();
    let mut json_serialization = Vec::new();
    let mut socket_send_call = Vec::new();

    // The record's sender clock is deliberately not used for window membership.
    // All comparisons below use the gateway receiver's own monotonic tick.
    for received in &snapshot.retained_records {
        if received.received_at_tick_ns < input.started_at_tick_ns
            || received.received_at_tick_ns > input.ended_at_tick_ns
            || received
                .record
                .get("record_type")
                .and_then(serde_json::Value::as_str)
                != Some("gateway_websocket_send")
        {
            continue;
        }
        let record = &received.record;
        gateway_send_records = gateway_send_records.saturating_add(1);
        if let Some(objective_record) = record.as_object() {
            if let Some(outcome) = objective_record
                .get("outcome")
                .and_then(serde_json::Value::as_str)
            {
                let count = outcomes.entry(outcome.to_owned()).or_default();
                *count = count.saturating_add(1);
            }
        }
        let client_id = record.get("client_id").and_then(serde_json::Value::as_u64);
        let publication_sequence = record
            .get("publication_sequence")
            .and_then(serde_json::Value::as_u64);
        if let (Some(client_id), Some(publication_sequence)) = (client_id, publication_sequence) {
            client_ids.insert(client_id);
            unique_publications.insert((client_id, publication_sequence));
        }
        repeated_publications = repeated_publications.saturating_add(u64::from(
            record
                .get("repeated_publication")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        ));
        repeated_source_observations = repeated_source_observations.saturating_add(u64::from(
            record
                .get("repeated_source_observation")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        ));
        publication_sequence_gaps = publication_sequence_gaps.saturating_add(
            record
                .get("publication_sequence_gap")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        );
        source_sequence_gaps = source_sequence_gaps.saturating_add(
            record
                .get("source_sequence_gap")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        );

        push_nonnegative_number(
            record,
            "publication_to_selection_ms",
            &mut publication_to_selection,
        );
        push_optional_nonnegative_number(
            record,
            "watch_wakeup_to_consume_ms",
            &mut watch_wakeup_to_consume,
        );
        push_nonnegative_number(record, "message_assembly_ms", &mut message_assembly);
        push_nonnegative_number(record, "json_serialization_ms", &mut json_serialization);
        push_optional_nonnegative_number(record, "socket_send_call_ms", &mut socket_send_call);
        if let Some(bytes) = record
            .get("payload_bytes")
            .and_then(serde_json::Value::as_u64)
        {
            payload_bytes.push(bytes as f64);
        }
        if detail_records.len() < maximum_samples {
            let mut record = record.clone();
            if let Some(object) = record.as_object_mut() {
                object.insert(
                    "audit_receiver_monotonic_ns".to_owned(),
                    serde_json::json!(received.received_at_tick_ns),
                );
            }
            detail_records.push(record);
        }
    }

    if let Some(footer) = &snapshot.producer_footer {
        if detail_records.len() < maximum_samples {
            let mut footer = footer.clone();
            if let Some(object) = footer.as_object_mut() {
                object.insert(
                    "audit_receiver_snapshot_tick_ns".to_owned(),
                    serde_json::json!(snapshot.measurement_clock_tick_ns),
                );
            }
            detail_records.push(footer);
        }
    }

    let run_complete = summarize_gateway_timing_receiver(snapshot, run_id).status == "complete";
    let status = if run_complete { "complete" } else { "partial" };
    let summary = serde_json::json!({
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "status": status,
        "gateway_run_id": run_id,
        "measurement_monotonic_window_ns": {
            "clock": "gateway_audit_receiver_monotonic",
            "clock_domain_id": snapshot.measurement_clock_domain_id,
            "started_at": input.started_at_tick_ns,
            "ended_at": input.ended_at_tick_ns,
        },
        "counts": {
            "gateway_send_records": gateway_send_records,
            "outcomes": outcomes,
            "unique_client_publications": unique_publications.len(),
            "connected_client_ids": client_ids.len(),
            "repeated_publications": repeated_publications,
            "repeated_source_observations": repeated_source_observations,
            "publication_sequence_gaps": publication_sequence_gaps,
            "source_sequence_gaps": source_sequence_gaps,
        },
        "latency_ms": {
            "publication_to_selection": distribution_summary(publication_to_selection),
            "watch_wakeup_to_consume": distribution_summary(watch_wakeup_to_consume),
            "message_assembly": distribution_summary(message_assembly),
            "json_serialization": distribution_summary(json_serialization),
            "socket_send_call": distribution_summary(socket_send_call),
        },
        "payload_bytes": distribution_summary(payload_bytes),
        "receiver_losses": snapshot.losses,
        "footer_complete": summarize_gateway_timing_receiver(snapshot, run_id).footer_complete,
        "footer": snapshot.producer_footer,
        "retained_record_count": snapshot.retained_records.len(),
        "retained_record_bytes": snapshot.retained_record_bytes,
        "maximum_retained_records": snapshot.maximum_retained_records,
        "maximum_retained_bytes": snapshot.maximum_retained_bytes,
        "publication_to_client_receipt": {
            "status": "unavailable",
            "reason": "client receipt time is not observed in this gateway process or receiver clock domain",
        },
        "physical_display_latency": "unavailable",
    });
    (summary, detail_records)
}

fn push_nonnegative_number(record: &serde_json::Value, key: &str, values: &mut Vec<f64>) {
    push_optional_nonnegative_number(record, key, values);
}

fn push_optional_nonnegative_number(record: &serde_json::Value, key: &str, values: &mut Vec<f64>) {
    if let Some(value) = record
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
    {
        values.push(value);
    }
}

fn distribution_summary(mut values: Vec<f64>) -> serde_json::Value {
    values.sort_by(f64::total_cmp);
    let count = values.len();
    if count == 0 {
        return serde_json::json!({
            "count": 0,
            "p50": null,
            "p95": null,
            "p99": null,
            "mean": null,
            "max": null,
        });
    }
    let percentile = |percent: usize| {
        let index = ((count * percent).div_ceil(100)).saturating_sub(1);
        values[index.min(count - 1)]
    };
    serde_json::json!({
        "count": count,
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "mean": values.iter().sum::<f64>() / count as f64,
        "max": values.last().copied(),
    })
}

fn write_live_timing_artifacts(
    run_directory: &Path,
    repetition: usize,
    summary: &serde_json::Value,
    detail_records: &[serde_json::Value],
) -> Result<(String, String)> {
    let summary_relative = format!("runs/run-{repetition:03}-live-pipeline-timing.json");
    let details_relative = format!("runs/run-{repetition:03}-live-pipeline-timing.jsonl");
    let summary_path = run_directory.join(&summary_relative);
    let summary_file = create_new_output(&summary_path, "live pipeline timing summary")?;
    let mut summary_writer = BufWriter::new(summary_file);
    serde_json::to_writer_pretty(
        &mut summary_writer,
        &serde_json::json!({
            "schema_version": RUN_REPORT_SCHEMA_VERSION,
            "tool": "billiards-performance-audit-run",
            "repetition": repetition,
            "summary": summary,
        }),
    )
    .context("could not serialize live timing window summary")?;
    summary_writer.write_all(b"\n")?;
    summary_writer.flush()?;

    let details_path = run_directory.join(&details_relative);
    let details_file = create_new_output(&details_path, "live pipeline timing details")?;
    let mut details_writer = BufWriter::new(details_file);
    for record in detail_records {
        serde_json::to_writer(&mut details_writer, record)
            .context("could not serialize live timing event")?;
        details_writer.write_all(b"\n")?;
    }
    details_writer.flush()?;
    Ok((summary_relative, details_relative))
}

fn write_gateway_timing_artifacts(
    run_directory: &Path,
    repetition: usize,
    summary: &serde_json::Value,
    detail_records: &[serde_json::Value],
) -> Result<(String, String)> {
    let summary_relative = format!("runs/run-{repetition:03}-gateway-websocket-timing.json");
    let details_relative = format!("runs/run-{repetition:03}-gateway-websocket-timing.jsonl");
    let summary_path = run_directory.join(&summary_relative);
    let summary_file = create_new_output(&summary_path, "gateway timing window summary")?;
    let mut summary_writer = BufWriter::new(summary_file);
    serde_json::to_writer_pretty(
        &mut summary_writer,
        &serde_json::json!({
            "schema_version": RUN_REPORT_SCHEMA_VERSION,
            "tool": "billiards-performance-audit-run",
            "repetition": repetition,
            "summary": summary,
        }),
    )
    .context("could not serialize gateway timing window summary")?;
    summary_writer.write_all(b"\n")?;
    summary_writer.flush()?;

    let details_path = run_directory.join(&details_relative);
    let details_file = create_new_output(&details_path, "gateway timing window details")?;
    let mut details_writer = BufWriter::new(details_file);
    for record in detail_records {
        serde_json::to_writer(&mut details_writer, record)
            .context("could not serialize gateway timing event")?;
        details_writer.write_all(b"\n")?;
    }
    details_writer.flush()?;
    Ok((summary_relative, details_relative))
}

fn write_window_artifacts(
    run_directory: &Path,
    repetition: usize,
    output: &StateReceiverOutput,
) -> Result<(String, String)> {
    let report_relative = format!("runs/run-{repetition:03}.json");
    let details_relative = format!("runs/run-{repetition:03}.jsonl");
    let runs_directory = run_directory.join("runs");
    fs::create_dir_all(&runs_directory)
        .with_context(|| format!("could not create {}", runs_directory.display()))?;

    let report_path = run_directory.join(&report_relative);
    let report_file = create_new_output(&report_path, "window report")?;
    let mut report_writer = BufWriter::new(report_file);
    serde_json::to_writer_pretty(
        &mut report_writer,
        &serde_json::json!({
            "schema_version": output.report.schema_version,
            "tool": output.report.tool,
            "report": output.report,
        }),
    )
    .context("could not serialize window report")?;
    report_writer.write_all(b"\n")?;
    report_writer.flush()?;

    let details_path = run_directory.join(&details_relative);
    let details_file = create_new_output(&details_path, "window details")?;
    let mut details_writer = BufWriter::new(details_file);
    for sample in &output.retained_samples {
        serde_json::to_writer(&mut details_writer, sample)
            .context("could not serialize retained state sample")?;
        details_writer.write_all(b"\n")?;
    }
    details_writer.flush()?;

    Ok((report_relative, details_relative))
}

fn write_report_artifacts(run_directory: &Path, report: &StateAuditRunReport) -> Result<()> {
    let summary_path = run_directory.join("summary.json");
    let summary_file = create_new_output(&summary_path, "summary report")?;
    let mut writer = BufWriter::new(summary_file);
    serde_json::to_writer_pretty(&mut writer, report).context("could not serialize summary")?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    let markdown_path = run_directory.join("report.md");
    let markdown_file = create_new_output(&markdown_path, "Markdown report")?;
    let mut markdown_writer = BufWriter::new(markdown_file);
    markdown_writer.write_all(render_markdown(report).as_bytes())?;
    markdown_writer.flush()?;
    Ok(())
}

fn render_markdown(report: &StateAuditRunReport) -> String {
    let mut output = format!(
        "# State delivery audit `{}`\n\nMode: **{}**. Started: {}. Finished: {}.\n\n",
        report.run_id, report.mode, report.started_at_utc, report.finished_at_utc,
    );
    if let Some(warmup) = &report.warmup {
        output.push_str(&format!(
            "Warm-up: {:.2}s requested, {:.2}s observed, {} state messages (excluded from measurements).\n\n",
            warmup.requested_duration_seconds,
            warmup.actual_duration_seconds,
            warmup.state_messages,
        ));
    }
    output.push_str(
        "| Run | State messages | Observed source frames | Incomplete states | Receive p50 / p95 / p99 (ms) |\n| ---: | ---: | ---: | ---: | ---: |\n",
    );
    for run in &report.measured_runs {
        let timing = &run.report.timing.local_state_receive_interval_ms;
        output.push_str(&format!(
            "| {} | {} | {} | {} | {} / {} / {} |\n",
            run.repetition,
            run.report.counts.state_messages,
            run.report.freshness.unique_observed_source_observations,
            run.report.completeness.incomplete_state_messages,
            optional_ms(timing.p50),
            optional_ms(timing.p95),
            optional_ms(timing.p99),
        ));
    }
    if let Some(live_timing) = &report.live_pipeline_timing {
        output.push_str(&format!(
            "\n## Live pipeline timing\n\nReceiver status: **{}**. {} records were received and {} remain in the bounded detail tail; receiver losses: malformed {}, oversized {}, queue-dropped {}, retention-evicted {}. Clock domain: `{}`.\n\n",
            live_timing.status,
            live_timing.stream_counts.valid_records,
            live_timing.retained_record_count,
            live_timing.losses.malformed_records,
            live_timing.losses.oversized_records,
            live_timing.losses.queue_dropped_records,
            live_timing.losses.retention_evictions,
            live_timing.measurement_clock_domain_id,
        ));
        output.push_str(
            "| Run | Timing status | Matched frames | Inference p95 (ms) | Frame elapsed p95 (ms) | Model exclusive p95 (ms) | Timing details |\n| ---: | :--- | ---: | ---: | ---: | ---: | :--- |\n",
        );
        for run in &report.measured_runs {
            let summary = run.live_timing_summary.as_ref();
            let status = summary
                .and_then(|summary| summary.get("status"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unavailable");
            let matched_frames = summary
                .and_then(|summary| summary.pointer("/state_timing_correlation/matched_frames"))
                .and_then(serde_json::Value::as_u64)
                .map_or_else(|| "unavailable".to_owned(), |count| count.to_string());
            let inference_p95 = summary
                .and_then(|summary| summary.pointer("/stage_durations_ms/inference/p95"))
                .and_then(serde_json::Value::as_f64)
                .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"));
            let frame_elapsed_p95 = summary
                .and_then(|summary| summary.pointer("/frame_timing/frame_elapsed_ms/p95"))
                .and_then(serde_json::Value::as_f64)
                .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"));
            let model_exclusive_p95 = summary
                .and_then(|summary| {
                    summary.pointer("/frame_timing/span_durations_ms/model_execute/exclusive/p95")
                })
                .and_then(serde_json::Value::as_f64)
                .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"));
            let detail_link = run
                .live_timing_details_path
                .as_deref()
                .map_or_else(|| "unavailable".to_owned(), |path| format!("`{path}`"));
            output.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                run.repetition,
                status,
                matched_frames,
                inference_p95,
                frame_elapsed_p95,
                model_exclusive_p95,
                detail_link,
            ));
        }
    }
    if let Some(gateway_timing) = &report.gateway_websocket_timing {
        output.push_str(&format!(
            "\n## Gateway WebSocket timing\n\nReceiver status: **{}**. Gateway run ID: `{}`. {} valid records were received and {} remain in the bounded detail tail; footer complete: **{}**. Receiver losses: malformed {}, oversized {}, queue-dropped {}, retention-evicted {}, transport errors {}. Measurement windows use receiver clock domain `{}`. Client receipt and physical display latency are unavailable.\n\n",
            gateway_timing.status,
            gateway_timing.run_id,
            gateway_timing.stream_counts.valid_records,
            gateway_timing.retained_record_count,
            gateway_timing.footer_complete,
            gateway_timing.losses.malformed_records,
            gateway_timing.losses.oversized_records,
            gateway_timing.losses.queue_dropped_records,
            gateway_timing.losses.retention_evictions,
            gateway_timing.losses.transport_errors,
            gateway_timing.measurement_clock_domain_id,
        ));
        output.push_str(
            "| Run | Status | Sends (sent / deadline / failed) | Selection p50 / p95 / p99 ms | Assembly p95 ms | JSON p95 ms | Socket-send p95 ms | Publication gaps / repeats | Source gaps / repeats | Payload mean / max bytes | Details |\n| ---: | :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- | :--- |\n",
        );
        for run in &report.measured_runs {
            let summary = run.gateway_timing_summary.as_ref();
            let lookup_u64 = |path: &str| {
                summary
                    .and_then(|value| value.pointer(path))
                    .and_then(serde_json::Value::as_u64)
            };
            let lookup_ms = |path: &str| {
                summary
                    .and_then(|value| value.pointer(path))
                    .and_then(serde_json::Value::as_f64)
                    .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"))
            };
            let status = summary
                .and_then(|value| value.get("status"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unavailable");
            let send_outcomes = format!(
                "{} / {} / {}",
                optional_count(lookup_u64("/counts/outcomes/sent")),
                optional_count(lookup_u64("/counts/outcomes/deadline_exceeded")),
                optional_count(Some(
                    lookup_u64("/counts/outcomes/send_failed")
                        .unwrap_or(0)
                        .saturating_add(
                            lookup_u64("/counts/outcomes/serialization_failed").unwrap_or(0),
                        ),
                )),
            );
            let selection = format!(
                "{} / {} / {}",
                lookup_ms("/latency_ms/publication_to_selection/p50"),
                lookup_ms("/latency_ms/publication_to_selection/p95"),
                lookup_ms("/latency_ms/publication_to_selection/p99"),
            );
            let publication = format!(
                "{} / {}",
                optional_count(lookup_u64("/counts/publication_sequence_gaps")),
                optional_count(lookup_u64("/counts/repeated_publications")),
            );
            let source = format!(
                "{} / {}",
                optional_count(lookup_u64("/counts/source_sequence_gaps")),
                optional_count(lookup_u64("/counts/repeated_source_observations")),
            );
            let payload = format!(
                "{} / {}",
                lookup_ms("/payload_bytes/mean"),
                lookup_ms("/payload_bytes/max"),
            );
            let details = run
                .gateway_timing_details_path
                .as_deref()
                .map_or("unavailable", |path| path);
            output.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` |\n",
                run.repetition,
                status,
                send_outcomes,
                selection,
                lookup_ms("/latency_ms/message_assembly/p95"),
                lookup_ms("/latency_ms/json_serialization/p95"),
                lookup_ms("/latency_ms/socket_send_call/p95"),
                publication,
                source,
                payload,
                details,
            ));
        }
    }
    output.push_str("\n## Provenance\n\n");
    output.push_str(&format!(
        "- Input: `{}`\n- Revision: `{}`\n- Working tree dirty: `{}`\n- Platform: `{} {}`\n- Logical CPUs: `{}`\n- Total memory: `{}` bytes\n- Maximum retained samples per run: `{}`\n",
        report.input.websocket_url,
        report.environment.repository_revision.as_deref().unwrap_or("unavailable"),
        report.environment.working_tree_dirty.map_or_else(|| "unavailable".to_owned(), |dirty| dirty.to_string()),
        report.environment.operating_system,
        report.environment.architecture,
        report.environment.logical_cpu_count.map_or_else(|| "unavailable".to_owned(), |count| count.to_string()),
        report.environment.total_memory_bytes.map_or_else(|| "unavailable".to_owned(), |bytes| bytes.to_string()),
        report.configuration.maximum_samples_per_run,
    ));
    if let Some(server) = &report.managed_server {
        output.push_str(&format!(
            "- Managed server: `{}` (PID `{}`), first observed frame `{}`, ready after {:.2}s across {} probes; shutdown `{}`\n",
            server.executable,
            server.process_id.map_or_else(|| "unavailable".to_owned(), |pid| pid.to_string()),
            server.first_observed_source_frame_id.as_deref().unwrap_or("unavailable"),
            server.startup_elapsed_seconds,
            server.readiness_probe_attempts,
            server.shutdown.as_ref().map_or("unavailable", |shutdown| shutdown.status),
        ));
        if let Some(live) = &server.live_provenance {
            output.push_str(&format!(
                "- Live camera profile: `{}`; provider selection: `{}`; model size: `{}`; confidence / IoU: `{:.4}` / `{:.4}`; ball/cue classes: `{}` / `{}`\n- Effective configuration SHA-256: `{}`\n- Capture settings SHA-256: `{}`\n- Detector model SHA-256: `{}`\n- Model manifest SHA-256: `{}`\n- Source checkpoint SHA-256: `{}`\n- Detector executable SHA-256: `{}`\n- Capture helper SHA-256: `{}`\n",
                live.camera_profile_id,
                live.provider_selection,
                live.model_image_size,
                live.confidence_threshold,
                live.iou_threshold,
                live.ball_class_id,
                live.cue_class_id.map_or_else(|| "unavailable".to_owned(), |id| id.to_string()),
                live.effective_configuration_sha256,
                live.capture_settings_sha256,
                live.detector_model_sha256,
                live.model_manifest_sha256,
                live.source_checkpoint_sha256,
                live.detector_executable_sha256,
                live.capture_helper_executable_sha256,
            ));
        }
    }
    output.push_str("\n## Limitations\n\n");
    for limitation in &report.limitations {
        output.push_str(&format!("- {limitation}\n"));
    }
    output
}

fn optional_ms(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.2}"))
}

fn optional_count(value: Option<u64>) -> String {
    value.map_or_else(|| "unavailable".to_owned(), |value| value.to_string())
}

pub(crate) fn collect_environment() -> RunEnvironment {
    let manifest_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let revision = git_output(manifest_directory, &["rev-parse", "--verify", "HEAD"]);
    let status = git_output(
        manifest_directory,
        &["status", "--porcelain", "--untracked-files=normal"],
    );
    let dirty = status.as_ref().map(|status| !status.is_empty());
    let fingerprint = status.map(|status| format!("{:x}", Sha256::digest(status.as_bytes())));
    let mut system = sysinfo::System::new();
    system.refresh_cpu_list(sysinfo::CpuRefreshKind::nothing().with_frequency());
    system.refresh_memory();
    let first_cpu = system.cpus().first();
    RunEnvironment {
        repository_revision: revision,
        working_tree_dirty: dirty,
        dirty_tree_status_sha256: fingerprint,
        operating_system: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        logical_cpu_count: std::thread::available_parallelism().ok().map(usize::from),
        processor_brand: first_cpu
            .map(|cpu| cpu.brand().trim().to_owned())
            .filter(|brand| !brand.is_empty()),
        processor_frequency_mhz: first_cpu
            .map(|cpu| cpu.frequency())
            .filter(|frequency| *frequency > 0),
        host_name: sysinfo::System::host_name(),
        kernel_version: sysinfo::System::kernel_version(),
        total_memory_bytes: Some(system.total_memory()),
        collector_version: env!("CARGO_PKG_VERSION"),
    }
}

fn git_output(manifest_directory: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_directory)
        .args(arguments)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_owned())
}

fn create_new_output(path: &Path, label: &str) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "could not create {label} output {} (existing files are never overwritten)",
                path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use super::{
        collect_environment, managed_live_provenance, render_markdown,
        GatewayWebSocketTimingAuditConfig, LivePipelineTimingAuditConfig, ManagedLiveProvenance,
        ManagedServerConfig, ManagedServerRunMetadata, ManagedServerShutdown, StateAuditRunConfig,
        StateAuditRunReport, DEFAULT_MEASURED_SECONDS, DEFAULT_REPETITIONS, DEFAULT_SOAK_SECONDS,
        DEFAULT_WARMUP_SECONDS, RUN_REPORT_SCHEMA_VERSION,
    };
    use crate::performance_audit_receiver::StateReceiverConfig;
    use serde_json::{json, Value};
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn config() -> StateAuditRunConfig {
        StateAuditRunConfig {
            url: "ws://127.0.0.1:8080/state".to_owned(),
            warmup: Duration::from_secs(DEFAULT_WARMUP_SECONDS),
            measured: Duration::from_secs(DEFAULT_MEASURED_SECONDS),
            repetitions: DEFAULT_REPETITIONS,
            soak: false,
            soak_duration: Duration::from_secs(DEFAULT_SOAK_SECONDS),
            output_root: super::default_output_root(),
            receiver: StateReceiverConfig::default(),
            managed_server: None,
            live_pipeline_timing_audit: None,
            gateway_websocket_timing_audit: None,
            sample_managed_live_process_resources: false,
        }
    }

    #[test]
    fn config_defaults_match_the_documented_repeatable_workflow() {
        let config = config();
        config.validate().unwrap();
        assert_eq!(config.warmup.as_secs(), 30);
        assert_eq!(config.measured.as_secs(), 120);
        assert_eq!(config.repetitions, 3);
        assert_eq!(config.soak_duration.as_secs(), 1_200);
        assert_eq!(
            config.receiver.maximum_samples,
            super::super::performance_audit_receiver::DEFAULT_MAXIMUM_SAMPLES
        );
    }

    #[test]
    fn default_audit_output_root_is_outside_the_source_tree() {
        let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("perception crate has a repository root");
        let output_root = super::default_output_root();

        assert!(!output_root.starts_with(repository_root));
        assert!(output_root.ends_with("billiards-performance-audits"));
    }

    #[test]
    fn config_rejects_empty_or_invalid_workloads_before_artifact_creation() {
        let mut invalid = config();
        invalid.repetitions = 0;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.url = "ws://user:secret@127.0.0.1:8080/state".to_owned();
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.soak_duration = Duration::ZERO;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.receiver.maximum_samples = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn managed_server_requires_loopback_and_positive_lifecycle_deadlines() {
        let mut valid = config();
        valid.managed_server = Some(ManagedServerConfig {
            executable: PathBuf::from("billiards-server"),
            arguments: vec!["--start-live".to_owned()],
            working_directory: None,
            startup_timeout: Duration::from_secs(60),
            readiness_probe_interval: Duration::from_millis(250),
            shutdown_timeout: Duration::from_secs(10),
        });
        assert!(valid.validate().is_ok());

        valid.url = "ws://192.0.2.10:8001/api/v1/websocket/ws".to_owned();
        assert!(valid.validate().is_err());

        valid.url = "ws://127.0.0.1:8001/api/v1/websocket/ws".to_owned();
        valid.managed_server.as_mut().unwrap().startup_timeout = Duration::ZERO;
        assert!(valid.validate().is_err());
    }

    #[test]
    fn live_timing_is_opt_in_and_requires_a_managed_live_server() {
        let mut invalid = config();
        invalid.live_pipeline_timing_audit = Some(LivePipelineTimingAuditConfig::default());
        assert!(invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires a managed server with --start-live"));

        invalid.managed_server = Some(ManagedServerConfig {
            executable: PathBuf::from("billiards-server"),
            arguments: vec!["--start-live".to_owned()],
            working_directory: None,
            startup_timeout: Duration::from_secs(60),
            readiness_probe_interval: Duration::from_millis(250),
            shutdown_timeout: Duration::from_secs(10),
        });
        assert!(invalid.validate().is_ok());

        invalid
            .live_pipeline_timing_audit
            .as_mut()
            .unwrap()
            .maximum_record_bytes = usize::MAX;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn gateway_timing_is_opt_in_and_independent_of_detector_timing() {
        let mut config = config();
        config.gateway_websocket_timing_audit = Some(GatewayWebSocketTimingAuditConfig::default());
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires a managed server with --start-live"));

        config.managed_server = Some(ManagedServerConfig {
            executable: PathBuf::from("billiards-server"),
            arguments: vec!["--start-live".to_owned()],
            working_directory: None,
            startup_timeout: Duration::from_secs(60),
            readiness_probe_interval: Duration::from_millis(250),
            shutdown_timeout: Duration::from_secs(10),
        });
        assert!(config.validate().is_ok());
        assert!(config.live_pipeline_timing_audit.is_none());

        config
            .gateway_websocket_timing_audit
            .as_mut()
            .unwrap()
            .maximum_record_bytes = usize::MAX;
        assert!(config.validate().is_err());
    }

    #[test]
    fn gateway_server_flags_include_its_own_listener_and_run_identity() {
        let mut arguments = Vec::new();
        let timing = GatewayWebSocketTimingAuditConfig::default();
        super::append_gateway_timing_server_arguments(
            &mut arguments,
            &timing,
            "127.0.0.1:45678".parse().unwrap(),
            "gateway-run-test",
        );
        assert_eq!(arguments[0], "--gateway-audit-tcp");
        assert_eq!(arguments[1], "127.0.0.1:45678");
        assert_eq!(arguments[arguments.len() - 2], "--gateway-audit-run-id");
        assert_eq!(arguments[arguments.len() - 1], "gateway-run-test");
        assert!(!arguments
            .iter()
            .any(|argument| argument == "--live-audit-tcp"));
    }

    #[test]
    fn gateway_window_summary_uses_receiver_clock_and_reports_outcomes_gaps_and_latency() {
        use super::{summarize_gateway_timing_window, GatewayWindowTimingInput};
        use crate::managed_live_audit::{
            ManagedLiveAuditLosses, ManagedLiveAuditSnapshot, ManagedLiveAuditStreamCounts,
            ReceivedAuditRecord,
        };

        let record =
            |outcome: &str, sequence: u64, tick: u64, selection_ms: f64| ReceivedAuditRecord {
                received_at_tick_ns: tick,
                record: json!({
                    "record_type": "gateway_websocket_send",
                    "monotonic_clock_domain_id": "gateway-process-clock",
                    "client_id": 7,
                    "publication_sequence": sequence,
                    "publication_to_selection_ms": selection_ms,
                    "watch_wakeup_to_consume_ms": 1.0,
                    "message_assembly_ms": 0.3,
                    "json_serialization_ms": 0.4,
                    "socket_send_call_ms": 2.0,
                    "payload_bytes": 256,
                    "outcome": outcome,
                    "repeated_publication": sequence == 9,
                    "repeated_source_observation": sequence == 9,
                    "publication_sequence_gap": if sequence == 9 { 2 } else { 0 },
                    "source_sequence_gap": if sequence == 9 {
                        json!(1)
                    } else {
                        serde_json::Value::Null
                    },
                    "engine_instance_id": "engine-a",
                    "source_session_id": "session-a",
                    "source_frame_id": "frame-a",
                }),
            };
        let snapshot = ManagedLiveAuditSnapshot {
            listener_address: "127.0.0.1:9001".parse().unwrap(),
            measurement_clock_domain_id: "gateway-receiver-clock".to_owned(),
            measurement_clock_tick_ns: 300,
            latest_producer_process_id: Some(901),
            producer_process_id_generation: 1,
            stream_counts: ManagedLiveAuditStreamCounts {
                framed_messages: 3,
                valid_records: 3,
                records_by_type: std::collections::BTreeMap::from([
                    ("gateway_websocket_send".to_owned(), 2),
                    ("footer".to_owned(), 1),
                ]),
            },
            producer_footer: Some(json!({
                "record_type": "footer",
                "export_status": "complete",
                "dropped_records": 0,
                "gateway_websocket_send_successes": 1,
                "gateway_websocket_send_deadlines": 1,
            })),
            losses: ManagedLiveAuditLosses::default(),
            retained_records: vec![
                record("sent", 8, 150, 10.0),
                record("deadline_exceeded", 9, 180, 30.0),
            ],
            retained_record_bytes: 700,
            maximum_retained_records: 100,
            maximum_retained_bytes: 4096,
        };
        let input = GatewayWindowTimingInput {
            repetition: 1,
            started_at_tick_ns: 100,
            ended_at_tick_ns: 200,
        };

        let (summary, details) =
            summarize_gateway_timing_window(&snapshot, &input, 10, "gateway-run");

        assert_eq!(summary["status"], "complete");
        assert_eq!(
            summary["measurement_monotonic_window_ns"]["clock_domain_id"],
            "gateway-receiver-clock"
        );
        assert_eq!(summary["counts"]["gateway_send_records"], 2);
        assert_eq!(summary["counts"]["outcomes"]["sent"], 1);
        assert_eq!(summary["counts"]["outcomes"]["deadline_exceeded"], 1);
        assert_eq!(summary["counts"]["publication_sequence_gaps"], 2);
        assert_eq!(summary["counts"]["repeated_publications"], 1);
        assert_eq!(summary["counts"]["source_sequence_gaps"], 1);
        assert_eq!(
            summary["latency_ms"]["publication_to_selection"]["p50"],
            10.0
        );
        assert_eq!(
            summary["latency_ms"]["publication_to_selection"]["p95"],
            30.0
        );
        assert_eq!(summary["payload_bytes"]["max"], 256.0);
        assert_eq!(
            summary["publication_to_client_receipt"]["status"],
            "unavailable"
        );
        assert_eq!(details.len(), 3, "two sends and the final producer footer");
        assert_eq!(details[0]["audit_receiver_monotonic_ns"], 150);
        assert_eq!(details[2]["record_type"], "footer");
    }

    #[tokio::test]
    async fn gateway_receiver_drains_a_distinct_stream_and_keeps_its_pid_and_clock() {
        use super::{summarize_gateway_timing_window, GatewayWindowTimingInput};
        use crate::managed_live_audit::{ManagedLiveAuditReceiver, ManagedLiveAuditReceiverConfig};
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpStream;

        async fn send_records(address: std::net::SocketAddr, records: &[serde_json::Value]) {
            let mut stream = TcpStream::connect(address).await.unwrap();
            for record in records {
                let bytes = serde_json::to_vec(record).unwrap();
                stream
                    .write_all(&(bytes.len() as u32).to_be_bytes())
                    .await
                    .unwrap();
                stream.write_all(&bytes).await.unwrap();
            }
            stream.shutdown().await.unwrap();
        }

        let detector = ManagedLiveAuditReceiver::bind(ManagedLiveAuditReceiverConfig::default())
            .await
            .unwrap();
        let gateway = ManagedLiveAuditReceiver::bind(ManagedLiveAuditReceiverConfig::default())
            .await
            .unwrap();
        let window_started = gateway.measurement_clock_tick_ns();
        send_records(
            detector.listener_address(),
            &[
                json!({"record_type":"frame", "process_id": 11}),
                json!({
                    "record_type":"footer",
                    "process_id": 11,
                    "export_status":"complete",
                    "dropped_records": 0
                }),
            ],
        )
        .await;
        send_records(
            gateway.listener_address(),
            &[
                json!({
                    "record_type":"gateway_websocket_send",
                    "process_id": 22,
                    "client_id": 4,
                    "publication_sequence": 12,
                    "publication_to_selection_ms": 4.5,
                    "watch_wakeup_to_consume_ms": null,
                    "message_assembly_ms": 0.2,
                    "json_serialization_ms": 0.3,
                    "socket_send_call_ms": 1.5,
                    "payload_bytes": 128,
                    "outcome":"sent",
                    "publication_sequence_gap": 0,
                    "source_sequence_gap": null,
                    "repeated_publication": false,
                    "repeated_source_observation": false
                }),
                json!({
                    "record_type":"footer",
                    "process_id": 22,
                    "export_status":"complete",
                    "dropped_records": 0
                }),
            ],
        )
        .await;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if gateway.snapshot().await.stream_counts.valid_records >= 2 {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "gateway records were not received"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let window_ended = gateway.measurement_clock_tick_ns();
        let detector_snapshot = detector.shutdown().await.unwrap();
        let gateway_snapshot = gateway.shutdown().await.unwrap();

        assert_eq!(detector_snapshot.latest_producer_process_id, Some(11));
        assert_eq!(
            detector_snapshot.stream_counts.records_by_type.get("frame"),
            Some(&1)
        );
        assert!(!detector_snapshot
            .stream_counts
            .records_by_type
            .contains_key("gateway_websocket_send"));
        assert_eq!(gateway_snapshot.latest_producer_process_id, Some(22));
        assert_eq!(
            gateway_snapshot
                .stream_counts
                .records_by_type
                .get("gateway_websocket_send"),
            Some(&1)
        );

        let (summary, details) = summarize_gateway_timing_window(
            &gateway_snapshot,
            &GatewayWindowTimingInput {
                repetition: 1,
                started_at_tick_ns: window_started,
                ended_at_tick_ns: window_ended,
            },
            10,
            "gateway-run-isolated",
        );
        assert_eq!(summary["status"], "complete");
        assert_eq!(
            summary["measurement_monotonic_window_ns"]["clock_domain_id"],
            gateway_snapshot.measurement_clock_domain_id
        );
        assert_eq!(summary["counts"]["gateway_send_records"], 1);
        assert_eq!(
            summary["latency_ms"]["publication_to_selection"]["p95"],
            4.5
        );
        assert_eq!(
            details.len(),
            2,
            "gateway details contain its own send plus footer"
        );
    }

    #[test]
    fn managed_live_resource_sampling_requires_timing_and_rejects_an_explicit_pid() {
        let mut invalid = config();
        invalid.sample_managed_live_process_resources = true;
        assert!(invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires --live-pipeline-timing-audit"));

        invalid.live_pipeline_timing_audit = Some(LivePipelineTimingAuditConfig::default());
        invalid.managed_server = Some(ManagedServerConfig {
            executable: PathBuf::from("billiards-server"),
            arguments: vec!["--start-live".to_owned()],
            working_directory: None,
            startup_timeout: Duration::from_secs(60),
            readiness_probe_interval: Duration::from_millis(250),
            shutdown_timeout: Duration::from_secs(10),
        });
        invalid.receiver.resource_sample_process_id = Some(123);
        assert!(invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("cannot be combined with an explicit resource process ID"));

        invalid.receiver.resource_sample_process_id = None;
        assert!(invalid.validate().is_ok());
    }

    #[test]
    fn managed_live_timing_joins_exact_window_records_to_observed_state_identity() {
        use super::{summarize_live_timing_window, WindowTimingInput};
        use crate::managed_live_audit::{
            ManagedLiveAuditLosses, ManagedLiveAuditSnapshot, ManagedLiveAuditStreamCounts,
            ReceivedAuditRecord,
        };

        let directory = tempfile::tempdir().unwrap();
        let state_details = directory.path().join("state.jsonl");
        std::fs::write(
            &state_details,
            format!(
                "{}\n",
                json!({
                    "observation_identity": "engine-a\u{1f}session-a\u{1f}frame-1",
                    "ball_positions_status": "observed",
                    "transport_sequence": 7,
                })
            ),
        )
        .unwrap();
        let record = json!({
            "record_type": "frame",
            "process_id": 501,
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "frame_id": "frame-1",
            "transport_sequence": 7,
            "durations_ms": { "inference": 2.5 },
            "frame_timing": {
                "frame_elapsed_ns": 5_000_000,
                "dropped_spans": 0,
                "invalid_clock_spans": 0,
                "spans": [{
                    "name": "model_execute",
                    "inclusive_elapsed_ns": 2_000_000,
                    "exclusive_elapsed_ns": 2_000_000
                }]
            },
        });
        let snapshot = ManagedLiveAuditSnapshot {
            listener_address: "127.0.0.1:9000".parse().unwrap(),
            measurement_clock_domain_id: "test-clock-domain".to_owned(),
            measurement_clock_tick_ns: 250,
            latest_producer_process_id: Some(501),
            producer_process_id_generation: 1,
            stream_counts: ManagedLiveAuditStreamCounts {
                framed_messages: 2,
                valid_records: 2,
                records_by_type: Default::default(),
            },
            producer_footer: Some(json!({
                "record_type": "footer",
                "process_id": 999,
                "export_status": "complete",
                "dropped_records": 0,
            })),
            losses: ManagedLiveAuditLosses::default(),
            retained_records: vec![ReceivedAuditRecord {
                received_at_tick_ns: 150,
                record: record.clone(),
            }],
            retained_record_bytes: 100,
            maximum_retained_records: 100,
            maximum_retained_bytes: 4096,
        };
        let input = WindowTimingInput {
            repetition: 1,
            started_at_tick_ns: 100,
            ended_at_tick_ns: 200,
            sampled_process_id: Some(501),
            process_resource_status: Some("available".to_owned()),
            details_path: state_details,
            state_trace_truncated: false,
        };

        let (summary, detail_records) = summarize_live_timing_window(
            &snapshot,
            &input,
            100,
            &snapshot.measurement_clock_domain_id,
            true,
        )
        .unwrap();

        assert_eq!(summary["status"], "complete");
        assert_eq!(summary["state_timing_correlation"]["matched_frames"], 1);
        assert_eq!(summary["stage_durations_ms"]["inference"]["count"], 1);
        assert_eq!(summary["frame_timing"]["frame_elapsed_ms"]["p95"], 5.0);
        assert_eq!(
            summary["frame_timing"]["span_durations_ms"]["model_execute"]["exclusive"]["p95"],
            2.0
        );
        assert_eq!(
            summary["measurement_monotonic_window_ns"]["clock_domain_id"],
            "test-clock-domain"
        );
        assert_eq!(detail_records.len(), 2, "frame plus post-window footer");
        assert_eq!(detail_records[0]["audit_receiver_monotonic_ns"], 150);
        assert_eq!(detail_records[1]["audit_receiver_monotonic_ns"], 250);
        assert_eq!(
            summary["managed_process_resource_sampling"]["status"],
            "complete"
        );
        assert_eq!(
            summary["managed_process_resource_sampling"]["sampled_process_id"],
            501
        );
        assert_eq!(
            summary["managed_process_resource_sampling"]["observed_process_ids"],
            json!([501])
        );
        assert_eq!(
            summary["managed_process_resource_sampling"]["restart_during_window"],
            false
        );

        let mut restarted_snapshot = snapshot.clone();
        let mut restarted_record = record;
        restarted_record["process_id"] = json!(502);
        restarted_snapshot
            .retained_records
            .push(ReceivedAuditRecord {
                received_at_tick_ns: 175,
                record: restarted_record,
            });
        let (restarted_summary, _) = summarize_live_timing_window(
            &restarted_snapshot,
            &input,
            100,
            &snapshot.measurement_clock_domain_id,
            true,
        )
        .unwrap();
        assert_eq!(
            restarted_summary["managed_process_resource_sampling"]["status"],
            "partial"
        );
        assert_eq!(
            restarted_summary["managed_process_resource_sampling"]["observed_process_ids"],
            json!([501, 502])
        );
        assert_eq!(
            restarted_summary["managed_process_resource_sampling"]["unsampled_process_ids"],
            json!([502])
        );
        assert_eq!(
            restarted_summary["managed_process_resource_sampling"]["restart_during_window"],
            true
        );
    }

    #[test]
    fn environment_reports_revision_and_dirty_state_when_git_is_available() {
        let environment = collect_environment();
        assert!(environment.repository_revision.is_some());
        assert!(environment.working_tree_dirty.is_some());
        assert!(environment.dirty_tree_status_sha256.is_some());
        assert!(environment.logical_cpu_count.is_some());
    }

    #[test]
    fn managed_live_provenance_hashes_model_runtime_and_effective_camera_profile() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let binary_directory = root.join("bin");
        let model_directory = root.join("models");
        std::fs::create_dir_all(&binary_directory).unwrap();
        std::fs::create_dir_all(&model_directory).unwrap();
        let server_executable = binary_directory.join("billiards-server");
        let detector_executable = binary_directory.join("billiards-perception-live");
        let capture_helper = binary_directory.join("billiards-perception-gstreamer-probe");
        let model_path = model_directory.join("detector.onnx");
        let model_manifest_path = PathBuf::from(format!("{}.manifest.json", model_path.display()));
        let checkpoint_path = model_directory.join("detector.pt");
        std::fs::write(&server_executable, b"gateway-test-binary").unwrap();
        std::fs::write(&detector_executable, b"detector-test-binary").unwrap();
        std::fs::write(&capture_helper, b"capture-helper-test-binary").unwrap();
        std::fs::write(&model_path, b"onnx-test-model").unwrap();
        std::fs::write(&model_manifest_path, b"model-manifest-test-content").unwrap();
        std::fs::write(&checkpoint_path, b"checkpoint-test-model").unwrap();

        let mut configuration: Value =
            serde_json::from_str(include_str!("../contracts/configuration/v1/defaults.json"))
                .unwrap();
        configuration["vision"]["camera"]["device_id"] = json!("http://127.0.0.1:8001/stream.mjpg");
        configuration["vision"]["camera"]["video_file_path"] = json!("");
        configuration["perception"]["rust_live"]["onnx_path"] = json!("models/detector.onnx");
        configuration["perception"]["rust_live"]["source_checkpoint_path"] =
            json!("models/detector.pt");
        configuration["perception"]["rust_live"]["executable_path"] =
            json!("bin/billiards-perception-live");
        configuration["perception"]["rust_live"]["capture_helper_executable_path"] =
            json!("bin/billiards-perception-gstreamer-probe");
        let configuration_path = root.join("config.json");
        std::fs::write(
            &configuration_path,
            serde_json::to_vec_pretty(&configuration).unwrap(),
        )
        .unwrap();
        let launch = ManagedServerConfig {
            executable: server_executable,
            arguments: vec![
                "--config".to_owned(),
                configuration_path.display().to_string(),
                "--project-root".to_owned(),
                root.display().to_string(),
                "--start-live".to_owned(),
            ],
            working_directory: Some(root.to_path_buf()),
            startup_timeout: Duration::from_secs(1),
            readiness_probe_interval: Duration::from_millis(20),
            shutdown_timeout: Duration::from_secs(1),
        };

        let provenance = managed_live_provenance(&launch).unwrap().unwrap();
        assert_eq!(
            provenance.detector_model_sha256,
            super::sha256_file(&model_path).unwrap()
        );
        assert_eq!(
            provenance.source_checkpoint_sha256,
            super::sha256_file(&checkpoint_path).unwrap()
        );
        assert_eq!(
            provenance.model_manifest_sha256,
            super::sha256_file(&model_manifest_path).unwrap()
        );
        assert_eq!(
            provenance.detector_executable_sha256,
            super::sha256_file(&detector_executable).unwrap()
        );
        assert_eq!(
            provenance.capture_helper_executable_sha256,
            super::sha256_file(&capture_helper).unwrap()
        );
        assert_eq!(provenance.provider_selection, "openvino");
        assert_eq!(provenance.ball_class_id, 1);
        assert_eq!(provenance.cue_class_id, Some(0));
        assert!(!provenance.effective_configuration_sha256.is_empty());
        assert!(!provenance.capture_settings_sha256.is_empty());
        assert!(!provenance.camera_profile_id.is_empty());
        let serialized = serde_json::to_string(&provenance).unwrap();
        assert!(!serialized.contains("http://127.0.0.1:8001/stream.mjpg"));
    }

    #[test]
    fn externally_managed_server_does_not_claim_live_model_provenance() {
        let launch = ManagedServerConfig {
            executable: PathBuf::from("server"),
            arguments: Vec::new(),
            working_directory: None,
            startup_timeout: Duration::from_secs(1),
            readiness_probe_interval: Duration::from_millis(20),
            shutdown_timeout: Duration::from_secs(1),
        };
        assert!(managed_live_provenance(&launch).unwrap().is_none());
    }

    #[test]
    fn markdown_discloses_accuracy_and_clock_limitations() {
        let report = StateAuditRunReport {
            schema_version: RUN_REPORT_SCHEMA_VERSION,
            tool: "billiards-performance-audit-run",
            run_id: "20261022-example".to_owned(),
            mode: "repeated",
            started_at_utc: "start".to_owned(),
            finished_at_utc: "finish".to_owned(),
            configuration: super::EffectiveRunConfiguration {
                warmup_seconds: 0.0,
                measured_seconds: 1.0,
                measurement_window_seconds: 1.0,
                repetitions: 1,
                soak: false,
                soak_seconds: 1_200.0,
                connect_timeout_seconds: 5.0,
                application_heartbeat_interval_seconds: 20.0,
                summary_interval_seconds: 5.0,
                maximum_message_bytes: 1_048_576,
                maximum_samples_per_run: 10_000,
                subscriber_delay_ms: 0,
                resource_process_id: None,
                resource_sample_interval_ms: 1_000,
                managed_live_process_resource_sampling: None,
                live_pipeline_timing_audit: None,
                gateway_websocket_timing_audit: Some(super::EffectiveLivePipelineTimingConfiguration {
                    queue_capacity: 128,
                    maximum_record_bytes: 65_536,
                    maximum_retained_records: 100,
                    maximum_retained_bytes: 4096,
                    connect_timeout_ms: 1000,
                    write_timeout_ms: 1000,
                }),
            },
            environment: collect_environment(),
            input: super::InputMetadata {
                kind: "websocket_state_stream",
                websocket_url: "ws://127.0.0.1:8080/state".to_owned(),
                content_sha256: None,
                content_hash_status: "unavailable_for_network_stream",
                detector_model: Some("sha256:model".to_owned()),
                detector_runtime: Some("cpu".to_owned()),
            },
            warmup: None,
            measured_runs: Vec::new(),
            live_pipeline_timing: None,
            gateway_websocket_timing: Some(super::ManagedGatewayWebSocketTimingSummary {
                status: "partial",
                run_id: "gateway-run-test".to_owned(),
                listener_address: "127.0.0.1:49152".to_owned(),
                measurement_clock_domain_id: "gateway-clock-test".to_owned(),
                stream_counts: crate::managed_live_audit::ManagedLiveAuditStreamCounts::default(),
                producer_footer: None,
                footer_complete: false,
                losses: crate::managed_live_audit::ManagedLiveAuditLosses::default(),
                retained_record_count: 0,
                retained_record_bytes: 0,
                maximum_retained_records: 100,
                maximum_retained_bytes: 4096,
                producer_process_id: Some(123),
            }),
            managed_server: Some(ManagedServerRunMetadata {
                executable: "billiards-server".to_owned(),
                process_id: Some(123),
                argument_count: 3,
                arguments_sha256: "arguments-hash".to_owned(),
                live_provenance: Some(ManagedLiveProvenance {
                    effective_configuration_sha256: "configuration-hash".to_owned(),
                    capture_settings_sha256: "capture-hash".to_owned(),
                    camera_profile_id: "camera-profile-test".to_owned(),
                    detector_model_sha256: "model-hash".to_owned(),
                    model_manifest_sha256: "manifest-hash".to_owned(),
                    source_checkpoint_sha256: "checkpoint-hash".to_owned(),
                    detector_executable_sha256: "detector-hash".to_owned(),
                    capture_helper_executable_sha256: "helper-hash".to_owned(),
                    provider_selection: "cpu".to_owned(),
                    model_image_size: 1280,
                    ball_class_id: 1,
                    cue_class_id: Some(0),
                    confidence_threshold: 0.3,
                    iou_threshold: 0.45,
                }),
                startup_timeout_seconds: 60.0,
                readiness_probe_interval_ms: 250,
                shutdown_timeout_seconds: 10.0,
                startup_elapsed_seconds: 2.0,
                readiness_probe_attempts: 3,
                first_observed_source_frame_id: Some("frame-1".to_owned()),
                shutdown: Some(ManagedServerShutdown {
                    status: "terminated_gracefully",
                    exit_code: Some(0),
                    signal: None,
                    forced: false,
                }),
            }),
            limitations: vec![
                "Detector accuracy is unavailable without temporally aligned ground-truth annotations.",
                "State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.",
            ],
        };
        let markdown = render_markdown(&report);
        assert!(markdown.contains("Detector accuracy is unavailable"));
        assert!(markdown.contains("unrelated clocks are not subtracted"));
        assert!(markdown.contains("Working tree dirty"));
        assert!(markdown.contains("camera-profile-test"));
        assert!(markdown.contains("Detector model SHA-256: `model-hash`"));
        assert!(markdown.contains("## Gateway WebSocket timing"));
        assert!(markdown.contains("Client receipt and physical display latency are unavailable"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_server_cleanup_stops_descendants_after_parent_exit() {
        use std::os::unix::fs::PermissionsExt;
        use std::time::Instant;

        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("server-exits-after-spawn.sh");
        let child_pid_path = directory.path().join("descendant.pid");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\n/bin/sleep 60 &\nprintf '%s\\n' \"$!\" > '{}'\nexit 0\n",
                child_pid_path.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let launch = ManagedServerConfig {
            executable: script,
            arguments: Vec::new(),
            working_directory: None,
            startup_timeout: Duration::from_secs(1),
            readiness_probe_interval: Duration::from_millis(20),
            shutdown_timeout: Duration::from_secs(1),
        };
        let mut running = super::spawn_managed_server(&launch, None, None).unwrap();
        let process_group_id = running.metadata.process_id.unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        let child_pid = loop {
            if let Ok(pid) = std::fs::read_to_string(&child_pid_path) {
                if running.child.try_wait().unwrap().is_some() {
                    break pid.trim().parse::<libc::pid_t>().unwrap();
                }
            }
            assert!(
                Instant::now() < deadline,
                "server did not exit after spawning its child"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        };

        let shutdown =
            super::stop_managed_server(&mut running.child, Some(process_group_id), &launch)
                .await
                .unwrap();

        assert_eq!(shutdown.status, "terminated_gracefully");
        assert!(!shutdown.forced);
        assert!(!super::process_group_exists(process_group_id as libc::pid_t).unwrap());
        // SAFETY: signal zero checks that the owned grandchild has been reaped
        // after its process group received SIGTERM.
        assert_eq!(unsafe { libc::kill(child_pid, 0) }, -1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_server_force_shutdown_kills_the_full_process_group() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("server-ignores-term.sh");
        let child_pid_path = directory.path().join("descendant.pid");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\ntrap '' TERM\n/bin/sleep 60 &\nprintf '%s\\n' \"$!\" > '{}'\nwait\n",
                child_pid_path.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let launch = ManagedServerConfig {
            executable: script,
            arguments: Vec::new(),
            working_directory: None,
            startup_timeout: Duration::from_secs(1),
            readiness_probe_interval: Duration::from_millis(20),
            shutdown_timeout: Duration::from_millis(150),
        };
        let mut running = super::spawn_managed_server(&launch, None, None).unwrap();
        let process_group_id = running.metadata.process_id.unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        while !child_pid_path.is_file() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "server did not start its child"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let shutdown =
            super::stop_managed_server(&mut running.child, Some(process_group_id), &launch)
                .await
                .unwrap();

        assert_eq!(shutdown.status, "force_killed");
        assert!(shutdown.forced);
        assert!(!super::process_group_exists(process_group_id as libc::pid_t).unwrap());
    }
}
