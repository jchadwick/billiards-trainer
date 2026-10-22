//! Standalone loopback receiver for recorded Rust detection-batch replay.

use std::fs::OpenOptions;
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process;
use std::time::Duration;

use anyhow::{Context, Result};
use billiards_perception::detection_batch_receiver::{serve_once, ReceiverConfig};
use clap::Parser;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Parser)]
#[command(
    name = "billiards-perception-receive-detection-batches",
    about = "Receive and validate one loopback detection-batch replay stream",
    long_about = "A bounded headless Stage 2 replay receiver. It records local monotonic receipt times and preserves valid batch JSON as JSONL for the retained equivalence scorer. It does not claim one-way latency from an unqualified cross-process clock."
)]
struct Arguments {
    /// Local loopback IP:port; use port 0 to request an available port.
    #[arg(long)]
    bind_address: SocketAddr,

    /// Number of framed detection batches expected from the producer.
    #[arg(long)]
    expected_batches: usize,

    /// Upper bound on expected batches and aggregate receiver memory.
    #[arg(long, default_value_t = 10_000)]
    maximum_batches: usize,

    /// Maximum length-prefixed JSON payload bytes per batch.
    #[arg(long)]
    maximum_message_bytes: usize,

    /// Deadline for accepting the producer connection.
    #[arg(long)]
    accept_timeout_ms: u64,

    /// Overall deadline for receiving and closing one producer session.
    #[arg(long)]
    session_timeout_ms: u64,

    /// Deadline for each frame header/body read and final sender close.
    #[arg(long)]
    read_timeout_ms: u64,

    /// Maximum per-batch receipt detail records retained in memory.
    #[arg(long, default_value_t = 10_000)]
    maximum_detail_records: usize,

    /// Create a new JSONL file containing each valid original batch payload.
    #[arg(long)]
    batches_jsonl: PathBuf,

    /// Optional path for a new versioned JSON receiver report.
    #[arg(long)]
    output_json: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    let arguments = Arguments::parse();
    match run(arguments).await {
        Ok(exit_code) => process::exit(exit_code),
        Err(error) => {
            eprintln!("detection batch receiver failed: {error:#}");
            process::exit(2);
        }
    }
}

async fn run(arguments: Arguments) -> Result<i32> {
    let config = ReceiverConfig {
        expected_batches: arguments.expected_batches,
        maximum_batches: arguments.maximum_batches,
        maximum_message_bytes: arguments.maximum_message_bytes,
        accept_timeout: Duration::from_millis(arguments.accept_timeout_ms),
        session_timeout: Duration::from_millis(arguments.session_timeout_ms),
        read_timeout: Duration::from_millis(arguments.read_timeout_ms),
        maximum_detail_records: arguments.maximum_detail_records,
    };
    config.validate()?;

    let mut jsonl_file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&arguments.batches_jsonl)
        .await
        .with_context(|| {
            format!(
                "could not create new batch JSONL file {}",
                arguments.batches_jsonl.display()
            )
        })?;

    let (mut report, _) = serve_once(arguments.bind_address, config, &mut jsonl_file, |address| {
        let event = serde_json::json!({
            "event": "ready",
            "address": address.to_string(),
        });
        println!("{event}");
    })
    .await?;
    jsonl_file
        .flush()
        .await
        .context("could not flush received batch JSONL")?;
    drop(jsonl_file);

    report
        .attach_jsonl_artifact(&arguments.batches_jsonl)
        .await?;
    let encoded = serde_json::to_string_pretty(&report)
        .context("could not serialize detection-batch receiver report")?;

    if let Some(path) = arguments.output_json {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("could not create new report {}", path.display()))?;
        output
            .write_all(encoded.as_bytes())
            .with_context(|| format!("could not write report {}", path.display()))?;
        output
            .write_all(b"\n")
            .with_context(|| format!("could not finish report {}", path.display()))?;
        output
            .flush()
            .with_context(|| format!("could not flush report {}", path.display()))?;
    }

    println!("{encoded}");
    Ok(if report.is_successful() { 0 } else { 1 })
}
