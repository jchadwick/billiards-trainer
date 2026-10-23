//! Headless state-stream receiver used by repeatable live performance audits.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use billiards_perception::performance_audit_receiver::{
    connect_and_collect, validate_websocket_url, StateReceiverConfig, StateReceiverOutput,
    DEFAULT_MAXIMUM_MESSAGE_BYTES, DEFAULT_MAXIMUM_SAMPLES,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    about = "Collect bounded delivery and freshness measurements from the live state WebSocket"
)]
struct Arguments {
    /// Anonymous plain-WebSocket endpoint, for example ws://127.0.0.1:8080/api/v1/websocket/ws.
    #[arg(long)]
    url: String,

    /// Collection duration. The default is suitable for smoke checks; use longer repeated runs for baselines.
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
    duration_seconds: u64,

    /// Deadline for connecting to the state endpoint.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    connect_timeout_seconds: u64,

    /// Interval for the API's JSON heartbeat. Separate from WebSocket control pings.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u64).range(1..))]
    application_heartbeat_interval_seconds: u64,

    /// How often to print a live terminal summary.
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..))]
    summary_interval_seconds: u64,

    /// Maximum accepted WebSocket message size.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_MESSAGE_BYTES, value_parser = clap::value_parser!(usize))]
    maximum_message_bytes: usize,

    /// Bound for retained details and percentile samples.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_SAMPLES, value_parser = clap::value_parser!(usize))]
    maximum_samples: usize,

    /// Optional process ID to sample for CPU and resident memory on a
    /// dedicated worker. Disabled when omitted.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    resource_process_id: Option<u32>,

    /// Sampling interval used when --resource-process-id is supplied.
    #[arg(long, default_value_t = 1000, value_parser = clap::value_parser!(u64).range(1..))]
    resource_sample_interval_ms: u64,

    /// Write the versioned report and bounded detail window as JSON. The file must not already exist.
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Optionally write retained state samples as JSON Lines. The file must not already exist.
    #[arg(long)]
    details_jsonl: Option<PathBuf>,
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

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    validate_websocket_url(&arguments.url)?;
    let config = StateReceiverConfig {
        duration: Duration::from_secs(arguments.duration_seconds),
        connect_timeout: Duration::from_secs(arguments.connect_timeout_seconds),
        application_heartbeat_interval: Duration::from_secs(
            arguments.application_heartbeat_interval_seconds,
        ),
        summary_interval: Duration::from_secs(arguments.summary_interval_seconds),
        maximum_message_bytes: arguments.maximum_message_bytes,
        maximum_samples: arguments.maximum_samples,
        resource_sample_process_id: arguments.resource_process_id,
        resource_sample_interval: Duration::from_millis(arguments.resource_sample_interval_ms),
        subscriber_delay: Duration::ZERO,
    };
    config.validate()?;
    if let (Some(report), Some(details)) = (&arguments.output_json, &arguments.details_jsonl) {
        if report == details {
            bail!("--output-json and --details-jsonl must refer to different paths");
        }
    }

    let mut report_file = arguments
        .output_json
        .as_deref()
        .map(|path| create_new_output(path, "JSON report"))
        .transpose()?;
    let mut details_file = arguments
        .details_jsonl
        .as_deref()
        .map(|path| create_new_output(path, "JSON Lines details"))
        .transpose()?;

    let output = connect_and_collect(&arguments.url, &config).await?;
    print_final_summary(&output);
    write_outputs(&output, report_file.take(), details_file.take())?;
    Ok(())
}

fn print_final_summary(output: &StateReceiverOutput) {
    let report = &output.report;
    let elapsed = report.elapsed_seconds;
    let states_per_second = rate(report.counts.state_messages, elapsed);
    let unique_per_second = rate(
        report.freshness.unique_observed_source_observations,
        elapsed,
    );
    println!(
        "audit complete: {:.2}s, {} state messages ({states_per_second:.2}/s), {} unique observed source frames ({unique_per_second:.2}/s), {} incomplete states, {} stale, {} source identities unavailable, termination={}",
        elapsed,
        report.counts.state_messages,
        report.freshness.unique_observed_source_observations,
        report.completeness.incomplete_state_messages,
        report.freshness.stale_state_messages,
        report.source_observations.messages_without_source_identity,
        report.termination_reason,
    );
    print_percentiles(
        "read-completion-to-publication age",
        &report.timing.read_completion_to_publication_age_ms,
    );
    print_percentiles(
        "local receive interval",
        &report.timing.local_state_receive_interval_ms,
    );
    if let Some(resources) = &report.process_resources {
        println!(
            "process resources: status={} pid={} samples={} retained={} cpu-per-core-p95={:?}% rss-max={:?}B",
            resources.status,
            resources.process_id,
            resources.sample_count,
            resources.retained_sample_count,
            resources.cpu_percent_per_core.p95,
            resources.resident_memory_bytes.maximum,
        );
    }
}

fn rate(count: u64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    }
}

fn print_percentiles(
    name: &str,
    summary: &billiards_perception::performance_audit_receiver::PercentileSummary,
) {
    match (summary.p50, summary.p95, summary.p99) {
        (Some(p50), Some(p95), Some(p99)) => println!(
            "{name}: p50={p50:.2}ms p95={p95:.2}ms p99={p99:.2}ms n={} retained={}{}",
            summary.sample_count,
            summary.retained_sample_count,
            if summary.truncated {
                " (window truncated)"
            } else {
                ""
            },
        ),
        _ => println!("{name}: unavailable (n={})", summary.sample_count),
    }
}

fn write_outputs(
    output: &StateReceiverOutput,
    report_file: Option<File>,
    details_file: Option<File>,
) -> Result<()> {
    if let Some(file) = report_file {
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, output)
            .context("could not serialize JSON audit report")?;
        writer
            .write_all(b"\n")
            .context("could not finish JSON audit report")?;
        writer
            .flush()
            .context("could not flush JSON audit report")?;
    } else {
        serde_json::to_writer_pretty(io::stdout().lock(), output)
            .context("could not write JSON audit report to stdout")?;
        println!();
    }

    if let Some(file) = details_file {
        let mut writer = BufWriter::new(file);
        for sample in &output.retained_samples {
            serde_json::to_writer(&mut writer, sample)
                .context("could not serialize a JSONL state sample")?;
            writer
                .write_all(b"\n")
                .context("could not write JSONL state sample")?;
        }
        writer
            .flush()
            .context("could not flush JSONL state samples")?;
    }
    Ok(())
}
