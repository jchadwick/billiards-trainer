//! Repeatable run orchestration and artifact writing for state-stream audits.
//!
//! The receiver measures one WebSocket window. This module owns the separate
//! audit-workflow contract: warm-up, repeated measured windows or one soak,
//! per-run detail files, environment provenance, and a versioned Markdown and
//! JSON report. It labels network-stream input content and detector accuracy as
//! unavailable; when it owns a live server, it also records hashes from that
//! server's effective configuration and model/runtime files.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::performance_audit_receiver::{
    connect_and_collect, connect_and_wait_for_first_fresh_observed_localization,
    validate_websocket_url, StateReceiverConfig, StateReceiverOutput, StateReceiverReport,
};

pub const DEFAULT_WARMUP_SECONDS: u64 = 30;
pub const DEFAULT_MEASURED_SECONDS: u64 = 120;
pub const DEFAULT_REPETITIONS: usize = 3;
pub const DEFAULT_SOAK_SECONDS: u64 = 1_200;
pub const DEFAULT_OUTPUT_ROOT: &str = "thoughts/performance-audits";
pub const RUN_REPORT_SCHEMA_VERSION: u8 = 3;
const MAXIMUM_REPETITIONS: usize = 1_000;
const PROVENANCE_HASH_BUFFER_SIZE: usize = 64 * 1024;

/// Settings for one repeatable state-only collection workflow.
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
    /// Optional application server child. Its first observed localization
    /// gates warm-up and measurements, and the child is always reaped before
    /// this command exits.
    pub managed_server: Option<ManagedServerConfig>,
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
    pub managed_server: Option<ManagedServerRunMetadata>,
    pub limitations: Vec<&'static str>,
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
    pub resource_process_id: Option<u32>,
    pub resource_sample_interval_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct RunEnvironment {
    pub repository_revision: Option<String>,
    pub working_tree_dirty: Option<bool>,
    pub dirty_tree_status_sha256: Option<String>,
    pub operating_system: &'static str,
    pub architecture: &'static str,
    pub logical_cpu_count: Option<usize>,
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
}

/// Run the configured warm-up and state-only measurement windows.
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
    let result = async {
        if let Some(launch) = &config.managed_server {
            let running = spawn_managed_server(launch)?;
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

        let mut report = if let Some(managed_child) = &mut child {
            run_windows_while_server_alive(
                config,
                &run_directory,
                &run_id,
                started_at_utc.clone(),
                environment,
                input,
                managed_child,
            )
            .await?
        } else {
            run_windows(
                config,
                &run_directory,
                &run_id,
                started_at_utc.clone(),
                environment,
                input,
            )
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

fn spawn_managed_server(config: &ManagedServerConfig) -> Result<RunningManagedServer> {
    let live_provenance = managed_live_provenance(config)?;
    let mut command = tokio::process::Command::new(&config.executable);
    command
        .args(&config.arguments)
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
    let argument_count = config.arguments.len();
    let mut argument_hasher = Sha256::new();
    for argument in &config.arguments {
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
    config: &StateAuditRunConfig,
    run_directory: &Path,
    run_id: &str,
    started_at_utc: String,
    environment: RunEnvironment,
    input: InputMetadata,
    child: &mut tokio::process::Child,
) -> Result<StateAuditRunReport> {
    tokio::select! {
        result = run_windows(config, run_directory, run_id, started_at_utc, environment, input) => result,
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

async fn run_windows(
    config: &StateAuditRunConfig,
    run_directory: &Path,
    run_id: &str,
    started_at_utc: String,
    environment: RunEnvironment,
    input: InputMetadata,
) -> Result<StateAuditRunReport> {
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
        let output = connect_and_collect(&config.url, &receiver)
            .await
            .with_context(|| format!("state audit measured run {repetition} failed"))?;
        if config.managed_server.is_some()
            && output.report.freshness.unique_observed_source_observations == 0
        {
            bail!("managed server produced no fresh observed state in measured run {repetition}");
        }
        let (report_path, details_path) =
            write_window_artifacts(run_directory, repetition, &output)?;
        measured_runs.push(MeasuredRunReference {
            repetition,
            report_path,
            details_path,
            report: output.report,
        });
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
        resource_process_id: config.receiver.resource_sample_process_id,
        resource_sample_interval_ms: config.receiver.resource_sample_interval.as_millis(),
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
        limitations.push(
            "Managed-server mode qualifies state delivery and process lifecycle; inference-process resource sampling is not yet included.",
        );
    }
    Ok(StateAuditRunReport {
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
        managed_server: None,
        limitations,
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

fn collect_environment() -> RunEnvironment {
    let manifest_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let revision = git_output(manifest_directory, &["rev-parse", "--verify", "HEAD"]);
    let status = git_output(
        manifest_directory,
        &["status", "--porcelain", "--untracked-files=normal"],
    );
    let dirty = status.as_ref().map(|status| !status.is_empty());
    let fingerprint = status.map(|status| format!("{:x}", Sha256::digest(status.as_bytes())));
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    RunEnvironment {
        repository_revision: revision,
        working_tree_dirty: dirty,
        dirty_tree_status_sha256: fingerprint,
        operating_system: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        logical_cpu_count: std::thread::available_parallelism().ok().map(usize::from),
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
        collect_environment, managed_live_provenance, render_markdown, ManagedLiveProvenance,
        ManagedServerConfig, ManagedServerRunMetadata, ManagedServerShutdown, StateAuditRunConfig,
        StateAuditRunReport, DEFAULT_MEASURED_SECONDS, DEFAULT_OUTPUT_ROOT, DEFAULT_REPETITIONS,
        DEFAULT_SOAK_SECONDS, DEFAULT_WARMUP_SECONDS, RUN_REPORT_SCHEMA_VERSION,
    };
    use crate::performance_audit_receiver::StateReceiverConfig;
    use serde_json::{json, Value};
    use std::path::PathBuf;
    use std::time::Duration;

    fn config() -> StateAuditRunConfig {
        StateAuditRunConfig {
            url: "ws://127.0.0.1:8080/state".to_owned(),
            warmup: Duration::from_secs(DEFAULT_WARMUP_SECONDS),
            measured: Duration::from_secs(DEFAULT_MEASURED_SECONDS),
            repetitions: DEFAULT_REPETITIONS,
            soak: false,
            soak_duration: Duration::from_secs(DEFAULT_SOAK_SECONDS),
            output_root: PathBuf::from(DEFAULT_OUTPUT_ROOT),
            receiver: StateReceiverConfig::default(),
            managed_server: None,
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
        assert_eq!(provenance.provider_selection, "cpu");
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
                resource_process_id: None,
                resource_sample_interval_ms: 1_000,
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
        let mut running = super::spawn_managed_server(&launch).unwrap();
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
        let mut running = super::spawn_managed_server(&launch).unwrap();
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
