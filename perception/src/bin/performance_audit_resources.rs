//! Small native helper for sampling a supervised live process during one run.
//!
//! The parent sends bounded `start` and `stop` JSON Lines controls over stdin.
//! The sampler itself uses a dedicated thread, so the parent remains free to
//! collect state and run the application without blocking on process metrics.

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use billiards_perception::process_resource_sampler::{
    ProcessResourceSampler, ProcessResourceSamplerConfig,
};
use clap::Parser;
use serde_json::{json, Value};

const MAXIMUM_CONTROL_LINE_BYTES: usize = 256;
const MAXIMUM_MESSAGE_LINE_BYTES: usize = 16 * 1024 * 1024;
const PROTOCOL_SCHEMA_VERSION: u8 = 2;
const DEFAULT_SAMPLE_INTERVAL_MS: u64 = 1_000;
const DEFAULT_MAXIMUM_SAMPLES: usize = 10_000;
const DEFAULT_STARTUP_TIMEOUT_MS: u64 = 5_000;
// The helper returns one bounded JSON summary line. Keep enough room for
// high-frequency qualification runs without allowing the final IPC message to
// grow without limit.
const MAXIMUM_SAMPLES: usize = 50_000;

#[derive(Debug, Parser)]
#[command(
    about = "Sample one process's CPU and resident memory over a bounded local control stream"
)]
struct Arguments {
    /// Process ID to sample. The process must already exist when sampling starts.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    process_id: u32,

    /// Sampling interval. Must respect sysinfo's minimum CPU refresh cadence.
    #[arg(long, default_value_t = DEFAULT_SAMPLE_INTERVAL_MS, value_parser = parse_interval_ms)]
    sample_interval_ms: u64,

    /// Maximum number of retained sample details.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_SAMPLES, value_parser = parse_maximum_samples)]
    maximum_samples: usize,

    /// Bound the wait for process identity and the sampler clock baseline.
    #[arg(long, default_value_t = DEFAULT_STARTUP_TIMEOUT_MS, value_parser = parse_positive_ms)]
    startup_timeout_ms: u64,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match run(arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("resource sampler failed: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn run(arguments: Arguments) -> Result<()> {
    let stdin = io::stdin();
    let mut input = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());

    write_message(
        &mut output,
        &json!({"kind": "ready", "schema_version": PROTOCOL_SCHEMA_VERSION}),
    )?;
    read_control(&mut input, "start")?;

    let mut sampler = ProcessResourceSampler::start(ProcessResourceSamplerConfig {
        process_id: arguments.process_id,
        sample_interval: Duration::from_millis(arguments.sample_interval_ms),
        maximum_samples: arguments.maximum_samples,
    })?;
    let readiness =
        sampler.wait_until_ready(Duration::from_millis(arguments.startup_timeout_ms))?;
    write_message(
        &mut output,
        &json!({
            "kind": "started",
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "process_id": readiness.process_id,
            "process_start_time_unix_seconds": readiness.process_start_time_unix_seconds,
        }),
    )?;

    read_control(&mut input, "stop")?;
    sampler.stop()?;
    write_message(
        &mut output,
        &json!({
            "kind": "summary",
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "summary": sampler.summary(),
        }),
    )?;
    Ok(())
}

fn read_control(reader: &mut impl BufRead, expected_kind: &str) -> Result<()> {
    let mut line = Vec::with_capacity(32);
    loop {
        let available = reader.fill_buf().context("could not read control input")?;
        if available.is_empty() {
            bail!("control input ended before {expected_kind:?}");
        }
        let line_end = available.iter().position(|byte| *byte == b'\n');
        let bytes_to_consume = line_end.map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(bytes_to_consume) > MAXIMUM_CONTROL_LINE_BYTES {
            bail!("control input exceeds {MAXIMUM_CONTROL_LINE_BYTES} bytes");
        }
        line.extend_from_slice(&available[..bytes_to_consume]);
        reader.consume(bytes_to_consume);
        if line_end.is_some() {
            break;
        }
    }
    while line.last().is_some_and(|byte| byte.is_ascii_whitespace()) {
        line.pop();
    }
    let value: Value = serde_json::from_slice(&line).context("invalid JSON control input")?;
    let object = value
        .as_object()
        .context("control input must be a JSON object")?;
    if object.len() != 1 || object.get("kind").and_then(Value::as_str) != Some(expected_kind) {
        bail!("expected exactly one control field `kind: {expected_kind}`");
    }
    Ok(())
}

fn write_message(writer: &mut impl Write, value: &Value) -> Result<()> {
    let mut message = serde_json::to_vec(value).context("could not serialize resource message")?;
    message.push(b'\n');
    if message.len() > MAXIMUM_MESSAGE_LINE_BYTES {
        bail!(
            "resource message exceeds the {}-byte line limit",
            MAXIMUM_MESSAGE_LINE_BYTES
        );
    }
    writer
        .write_all(&message)
        .context("could not finish resource message")?;
    writer.flush().context("could not flush resource message")
}

fn parse_interval_ms(value: &str) -> std::result::Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| "sample interval must be an integer in milliseconds".to_owned())?;
    if parsed == 0 || Duration::from_millis(parsed) < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
        return Err(format!(
            "sample interval must be at least {} ms",
            sysinfo::MINIMUM_CPU_UPDATE_INTERVAL.as_millis()
        ));
    }
    Ok(parsed)
}

fn parse_maximum_samples(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "maximum samples must be a positive integer".to_owned())?;
    if !(1..=MAXIMUM_SAMPLES).contains(&parsed) {
        return Err(format!(
            "maximum samples must be between 1 and {MAXIMUM_SAMPLES}"
        ));
    }
    Ok(parsed)
}

fn parse_positive_ms(value: &str) -> std::result::Result<u64, String> {
    let parsed = value
        .parse::<u64>()
        .map_err(|_| "startup timeout must be an integer in milliseconds".to_owned())?;
    if parsed == 0 {
        return Err("startup timeout must be positive".to_owned());
    }
    Ok(parsed)
}
