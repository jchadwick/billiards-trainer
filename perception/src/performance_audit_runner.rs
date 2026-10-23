//! Repeatable run orchestration and artifact writing for state-stream audits.
//!
//! The receiver measures one WebSocket window. This module owns the separate
//! audit-workflow contract: warm-up, repeated measured windows or one soak,
//! per-run detail files, environment provenance, and a versioned Markdown and
//! JSON report. It intentionally labels network-stream inputs as unavailable
//! for content hashing and detector accuracy; a delivery receiver cannot infer
//! those facts from state messages alone.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::performance_audit_receiver::{
    connect_and_collect, validate_websocket_url, StateReceiverConfig, StateReceiverOutput,
    StateReceiverReport,
};

pub const DEFAULT_WARMUP_SECONDS: u64 = 30;
pub const DEFAULT_MEASURED_SECONDS: u64 = 120;
pub const DEFAULT_REPETITIONS: usize = 3;
pub const DEFAULT_SOAK_SECONDS: u64 = 1_200;
pub const DEFAULT_OUTPUT_ROOT: &str = "thoughts/performance-audits";
pub const RUN_REPORT_SCHEMA_VERSION: u8 = 1;
const MAXIMUM_REPETITIONS: usize = 1_000;

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
}

impl StateAuditRunConfig {
    pub fn validate(&self) -> Result<()> {
        validate_websocket_url(&self.url)?;
        self.receiver.validate()?;
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
    pub limitations: Vec<&'static str>,
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
    let input = InputMetadata {
        kind: "websocket_state_stream",
        websocket_url: config.url.clone(),
        content_sha256: None,
        content_hash_status: "unavailable_for_network_stream",
        detector_model: None,
        detector_runtime: None,
    };

    let result = run_windows(
        config,
        &run_directory,
        &run_id,
        started_at_utc.clone(),
        environment,
        input,
    )
    .await;

    match result {
        Ok(report) => {
            write_report_artifacts(&run_directory, &report)?;
            Ok(run_directory)
        }
        Err(error) => {
            let failure_path = run_directory.join("failure.json");
            if let Ok(file) = create_new_output(&failure_path, "failure report") {
                let failure = serde_json::json!({
                    "schema_version": RUN_REPORT_SCHEMA_VERSION,
                    "tool": "billiards-performance-audit-run",
                    "run_id": run_id,
                    "error": format!("{error:#}"),
                    "failed_at_utc": Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
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
        limitations: vec![
            "This receiver measures state publication and network delivery; it does not measure camera exposure-to-state latency.",
            "Detector accuracy is unavailable without temporally aligned ground-truth annotations.",
            "A network stream has no input-content hash, and the receiver cannot identify the detector model or runtime.",
            "State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.",
        ],
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
        collect_environment, render_markdown, StateAuditRunConfig, StateAuditRunReport,
        DEFAULT_MEASURED_SECONDS, DEFAULT_OUTPUT_ROOT, DEFAULT_REPETITIONS, DEFAULT_SOAK_SECONDS,
        DEFAULT_WARMUP_SECONDS, RUN_REPORT_SCHEMA_VERSION,
    };
    use crate::performance_audit_receiver::StateReceiverConfig;
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
    fn environment_reports_revision_and_dirty_state_when_git_is_available() {
        let environment = collect_environment();
        assert!(environment.repository_revision.is_some());
        assert!(environment.working_tree_dirty.is_some());
        assert!(environment.dirty_tree_status_sha256.is_some());
        assert!(environment.logical_cpu_count.is_some());
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
                detector_model: None,
                detector_runtime: None,
            },
            warmup: None,
            measured_runs: Vec::new(),
            limitations: vec![
                "Detector accuracy is unavailable without temporally aligned ground-truth annotations.",
                "State receipt intervals and source-reported ages use their documented clock domains; unrelated clocks are not subtracted.",
            ],
        };
        let markdown = render_markdown(&report);
        assert!(markdown.contains("Detector accuracy is unavailable"));
        assert!(markdown.contains("unrelated clocks are not subtracted"));
        assert!(markdown.contains("Working tree dirty"));
    }
}
