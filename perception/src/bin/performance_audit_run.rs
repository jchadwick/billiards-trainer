//! Repeatable state-only audit workflow and bounded artifact writer.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use billiards_perception::performance_audit_receiver::{
    StateReceiverConfig, DEFAULT_MAXIMUM_MESSAGE_BYTES, DEFAULT_MAXIMUM_SAMPLES,
};
use billiards_perception::performance_audit_runner::{
    run_state_audit, LivePipelineTimingAuditConfig, ManagedServerConfig, StateAuditRunConfig,
    DEFAULT_LIVE_AUDIT_CONNECT_TIMEOUT, DEFAULT_LIVE_AUDIT_MAX_RECORD_BYTES,
    DEFAULT_LIVE_AUDIT_QUEUE_CAPACITY, DEFAULT_LIVE_AUDIT_RETAINED_BYTES,
    DEFAULT_LIVE_AUDIT_RETAINED_RECORDS, DEFAULT_LIVE_AUDIT_WRITE_TIMEOUT,
    DEFAULT_MEASURED_SECONDS, DEFAULT_OUTPUT_ROOT, DEFAULT_REPETITIONS, DEFAULT_SOAK_SECONDS,
    DEFAULT_WARMUP_SECONDS,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Run repeatable warm-up, repeated, or soak state-delivery audits")]
struct Arguments {
    /// Anonymous plain-WebSocket endpoint for state updates.
    #[arg(long)]
    url: String,

    /// Excluded warm-up duration before measured windows; zero disables warm-up.
    #[arg(long, default_value_t = DEFAULT_WARMUP_SECONDS)]
    warmup_seconds: u64,

    /// Duration of each ordinary measured repetition.
    #[arg(long, default_value_t = DEFAULT_MEASURED_SECONDS, value_parser = clap::value_parser!(u64).range(1..))]
    duration_seconds: u64,

    /// Number of measured repetitions (1 to 1,000).
    #[arg(long, default_value_t = DEFAULT_REPETITIONS as u64, value_parser = clap::value_parser!(u64).range(1..=1000))]
    repetitions: u64,

    /// Run one bounded soak instead of ordinary repetitions.
    #[arg(long)]
    soak: bool,

    /// Soak duration when --soak is selected.
    #[arg(long, default_value_t = DEFAULT_SOAK_SECONDS, value_parser = clap::value_parser!(u64).range(1..))]
    soak_seconds: u64,

    /// Deadline for connecting to the state endpoint.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    connect_timeout_seconds: u64,

    /// Interval for the application's JSON heartbeat.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u64).range(1..))]
    application_heartbeat_interval_seconds: u64,

    /// How often to print a live receiver summary.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    summary_interval_seconds: u64,

    /// Maximum accepted WebSocket message size.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_MESSAGE_BYTES, value_parser = clap::value_parser!(usize))]
    maximum_message_bytes: usize,

    /// Maximum retained percentile and detail samples per measured window.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_SAMPLES as u64, value_parser = clap::value_parser!(u64).range(1..=1000000))]
    maximum_samples: u64,

    /// Optional process ID for separate CPU and resident-memory sampling.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    resource_process_id: Option<u32>,

    /// Resource sampling interval when a process ID is supplied.
    #[arg(long, default_value_t = 1000, value_parser = clap::value_parser!(u64).range(1..))]
    resource_sample_interval_ms: u64,

    /// Root directory for uniquely named run folders.
    #[arg(long, default_value = DEFAULT_OUTPUT_ROOT)]
    output_root: PathBuf,

    /// Start this local application-server executable before waiting for a fresh state.
    #[arg(long, requires = "url")]
    server_executable: Option<PathBuf>,

    /// One literal argument forwarded to the managed server. Repeat as needed.
    #[arg(
        long = "server-argument",
        allow_hyphen_values = true,
        requires = "server_executable"
    )]
    server_arguments: Vec<String>,

    /// Working directory for the managed server process.
    #[arg(long, requires = "server_executable")]
    server_working_directory: Option<PathBuf>,

    /// Maximum time to wait for the first fresh observed source state.
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u64).range(1..), requires = "server_executable")]
    startup_timeout_seconds: u64,

    /// Maximum interval between readiness WebSocket probes.
    #[arg(long, default_value_t = 250, value_parser = clap::value_parser!(u64).range(1..), requires = "server_executable")]
    readiness_probe_interval_ms: u64,

    /// Grace period for the managed server to shut down after the audit.
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u64).range(1..), requires = "server_executable")]
    server_shutdown_timeout_seconds: u64,

    /// Capture correlated detector timing records from a managed live server.
    #[arg(long, requires = "server_executable")]
    live_pipeline_timing_audit: bool,

    /// Bounded receiver and producer queue capacity for live timing records.
    #[arg(long, default_value_t = DEFAULT_LIVE_AUDIT_QUEUE_CAPACITY as u64, value_parser = clap::value_parser!(u64).range(1..=4096))]
    live_audit_queue_capacity: u64,

    /// Maximum framed live timing record accepted from the detector.
    #[arg(long, default_value_t = DEFAULT_LIVE_AUDIT_MAX_RECORD_BYTES as u64, value_parser = clap::value_parser!(u64).range(1024..=1048576))]
    live_audit_max_record_bytes: u64,

    /// Maximum detailed timing records retained for a complete managed run.
    #[arg(long, default_value_t = DEFAULT_LIVE_AUDIT_RETAINED_RECORDS as u64, value_parser = clap::value_parser!(u64).range(1..=1000000))]
    live_audit_retained_records: u64,

    /// Maximum bytes retained for detailed timing records.
    #[arg(long, default_value_t = DEFAULT_LIVE_AUDIT_RETAINED_BYTES as u64, value_parser = clap::value_parser!(u64).range(1..=536870912))]
    live_audit_retained_bytes: u64,

    /// Producer connection deadline for the loopback timing side channel.
    #[arg(long, default_value_t = DEFAULT_LIVE_AUDIT_CONNECT_TIMEOUT.as_millis() as u64, value_parser = clap::value_parser!(u64).range(1..))]
    live_audit_connect_timeout_ms: u64,

    /// Producer write deadline for the loopback timing side channel.
    #[arg(long, default_value_t = DEFAULT_LIVE_AUDIT_WRITE_TIMEOUT.as_millis() as u64, value_parser = clap::value_parser!(u64).range(1..))]
    live_audit_write_timeout_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let managed_server = arguments
        .server_executable
        .map(|executable| ManagedServerConfig {
            executable,
            arguments: arguments.server_arguments,
            working_directory: arguments.server_working_directory,
            startup_timeout: Duration::from_secs(arguments.startup_timeout_seconds),
            readiness_probe_interval: Duration::from_millis(arguments.readiness_probe_interval_ms),
            shutdown_timeout: Duration::from_secs(arguments.server_shutdown_timeout_seconds),
        });
    let live_pipeline_timing_audit = if arguments.live_pipeline_timing_audit {
        Some(LivePipelineTimingAuditConfig {
            queue_capacity: usize::try_from(arguments.live_audit_queue_capacity)?,
            maximum_record_bytes: usize::try_from(arguments.live_audit_max_record_bytes)?,
            maximum_retained_records: usize::try_from(arguments.live_audit_retained_records)?,
            maximum_retained_bytes: usize::try_from(arguments.live_audit_retained_bytes)?,
            connect_timeout: Duration::from_millis(arguments.live_audit_connect_timeout_ms),
            write_timeout: Duration::from_millis(arguments.live_audit_write_timeout_ms),
        })
    } else {
        None
    };
    let config = StateAuditRunConfig {
        url: arguments.url,
        warmup: Duration::from_secs(arguments.warmup_seconds),
        measured: Duration::from_secs(arguments.duration_seconds),
        repetitions: arguments.repetitions as usize,
        soak: arguments.soak,
        soak_duration: Duration::from_secs(arguments.soak_seconds),
        output_root: arguments.output_root,
        receiver: StateReceiverConfig {
            duration: Duration::from_secs(arguments.duration_seconds),
            connect_timeout: Duration::from_secs(arguments.connect_timeout_seconds),
            application_heartbeat_interval: Duration::from_secs(
                arguments.application_heartbeat_interval_seconds,
            ),
            summary_interval: Duration::from_secs(arguments.summary_interval_seconds),
            maximum_message_bytes: arguments.maximum_message_bytes,
            maximum_samples: arguments.maximum_samples as usize,
            resource_sample_process_id: arguments.resource_process_id,
            resource_sample_interval: Duration::from_millis(arguments.resource_sample_interval_ms),
        },
        managed_server,
        live_pipeline_timing_audit,
    };
    let directory = run_state_audit(&config).await?;
    println!("audit artifacts: {}", directory.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Arguments;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn cli_accepts_a_managed_server_with_literal_forwarded_arguments() {
        let arguments = Arguments::try_parse_from([
            "billiards-performance-audit-run",
            "--url",
            "ws://127.0.0.1:8001/api/v1/websocket/ws",
            "--server-executable",
            "target/debug/billiards-server",
            "--server-argument=--config",
            "--server-argument=config.json",
            "--server-argument=--start-live",
        ])
        .unwrap();

        assert_eq!(
            arguments.server_executable.unwrap(),
            PathBuf::from("target/debug/billiards-server")
        );
        assert_eq!(
            arguments.server_arguments,
            ["--config", "config.json", "--start-live"]
        );
    }

    #[test]
    fn cli_rejects_server_arguments_without_a_managed_server() {
        let result = Arguments::try_parse_from([
            "billiards-performance-audit-run",
            "--url",
            "ws://127.0.0.1:8001/api/v1/websocket/ws",
            "--server-argument=--start-live",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_accepts_live_timing_opt_in_for_a_managed_server() {
        let arguments = Arguments::try_parse_from([
            "billiards-performance-audit-run",
            "--url",
            "ws://127.0.0.1:8080/state",
            "--server-executable",
            "target/debug/billiards-server",
            "--server-argument=--start-live",
            "--live-pipeline-timing-audit",
            "--live-audit-queue-capacity",
            "128",
            "--live-audit-max-record-bytes",
            "8192",
        ])
        .unwrap();

        assert!(arguments.live_pipeline_timing_audit);
        assert_eq!(arguments.live_audit_queue_capacity, 128);
        assert_eq!(arguments.live_audit_max_record_bytes, 8192);
    }
}
