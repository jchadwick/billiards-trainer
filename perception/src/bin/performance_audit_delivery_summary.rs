//! Bounded JSON Lines adapter for native delivery-latency audit summaries.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use billiards_perception::delivery_latency_summary::summarize;
use clap::Parser;
use serde_json::Value;

const DEFAULT_MAXIMUM_EVENTS: usize = 200_000;
const MAXIMUM_EVENTS: usize = 5_000_000;
const DEFAULT_MAXIMUM_LINE_BYTES: usize = 1_048_576;
const MAXIMUM_LINE_BYTES: usize = 16_777_216;
const DEFAULT_MAXIMUM_INPUT_BYTES: usize = 134_217_728;
const MAXIMUM_INPUT_BYTES: usize = 1_073_741_824;

#[derive(Debug, Parser)]
#[command(about = "Summarize delivery latency from bounded client/server JSON Lines events")]
struct Arguments {
    /// Maximum client and server event objects accepted from stdin.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_EVENTS, value_parser = parse_maximum_events)]
    maximum_events: usize,

    /// Maximum bytes accepted for any input line, including its newline.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_LINE_BYTES, value_parser = parse_maximum_line_bytes)]
    maximum_line_bytes: usize,

    /// Maximum total bytes accepted from stdin.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_INPUT_BYTES, value_parser = parse_maximum_input_bytes)]
    maximum_input_bytes: usize,
}

struct Input {
    deadline_budget_ms: f64,
    server_clock_domain_id: Option<String>,
    clock_name: String,
    client_events: Vec<Value>,
    server_events: Vec<Value>,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match run(arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("delivery latency summary failed: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn run(arguments: Arguments) -> Result<()> {
    validate_limits(
        arguments.maximum_events,
        arguments.maximum_line_bytes,
        arguments.maximum_input_bytes,
    )?;
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut parsed = None;
    let mut line_number = 0_u64;
    let mut input_bytes = 0_usize;
    let mut event_count = 0_usize;
    while let Some(line) = read_bounded_line(&mut input, arguments.maximum_line_bytes)
        .with_context(|| {
            format!(
                "could not read JSON Lines input at line {}",
                line_number + 1
            )
        })?
    {
        line_number = line_number.saturating_add(1);
        input_bytes = input_bytes
            .checked_add(line.len())
            .context("JSON Lines input byte counter overflowed")?;
        if input_bytes > arguments.maximum_input_bytes {
            bail!(
                "input exceeds maximum byte count {}",
                arguments.maximum_input_bytes
            );
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let record: Value = serde_json::from_slice(&line)
            .with_context(|| format!("invalid JSON record at line {line_number}"))?;
        let record = record
            .as_object()
            .context("each JSON Lines record must be an object")?;
        let kind = record
            .get("kind")
            .and_then(Value::as_str)
            .context("each record must have a string kind")?;
        if parsed.is_none() {
            if kind != "config" {
                bail!("first record must be the config record");
            }
            let budget = record
                .get("deadline_budget_ms")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .context("config deadline_budget_ms must be finite and non-negative")?;
            let clock_name = record
                .get("clock_name")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .context("config clock_name must be a non-empty string")?
                .to_owned();
            let server_clock_domain_id = match record.get("server_clock_domain_id") {
                Some(Value::Null) | None => None,
                Some(Value::String(value)) => Some(value.clone()),
                _ => bail!("config server_clock_domain_id must be a string or null"),
            };
            parsed = Some(Input {
                deadline_budget_ms: budget,
                server_clock_domain_id,
                clock_name,
                client_events: Vec::new(),
                server_events: Vec::new(),
            });
            continue;
        }
        let input = parsed.as_mut().context("missing config record")?;
        if kind == "config" {
            bail!("config record may only appear once, as the first record");
        }
        let event = record
            .get("event")
            .filter(|value| value.is_object())
            .context("event record must include an object event")?
            .clone();
        event_count = event_count
            .checked_add(1)
            .context("event count overflowed")?;
        if event_count > arguments.maximum_events {
            bail!(
                "input exceeds maximum event count {}",
                arguments.maximum_events
            );
        }
        match kind {
            "client" => input.client_events.push(event),
            "server" => input.server_events.push(event),
            _ => bail!("event record kind must be client or server"),
        }
    }

    let input = parsed.context("input is missing the config record")?;
    let summary = summarize(
        &input.client_events,
        &input.server_events,
        input.deadline_budget_ms,
        input.server_clock_domain_id.as_deref(),
        &input.clock_name,
    );
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &summary).context("could not serialize summary")?;
    output
        .write_all(b"\n")
        .context("could not finish summary output")?;
    output.flush().context("could not flush summary output")?;
    Ok(())
}

fn validate_limits(
    maximum_events: usize,
    maximum_line_bytes: usize,
    maximum_input_bytes: usize,
) -> Result<()> {
    if !(1..=MAXIMUM_EVENTS).contains(&maximum_events) {
        bail!("--maximum-events must be between 1 and {MAXIMUM_EVENTS}");
    }
    if !(1..=MAXIMUM_LINE_BYTES).contains(&maximum_line_bytes) {
        bail!("--maximum-line-bytes must be between 1 and {MAXIMUM_LINE_BYTES}");
    }
    if !(1..=MAXIMUM_INPUT_BYTES).contains(&maximum_input_bytes) {
        bail!("--maximum-input-bytes must be between 1 and {MAXIMUM_INPUT_BYTES}");
    }
    Ok(())
}

fn parse_maximum_events(value: &str) -> std::result::Result<usize, String> {
    parse_bounded_usize(value, MAXIMUM_EVENTS, "--maximum-events")
}

fn parse_maximum_line_bytes(value: &str) -> std::result::Result<usize, String> {
    parse_bounded_usize(value, MAXIMUM_LINE_BYTES, "--maximum-line-bytes")
}

fn parse_maximum_input_bytes(value: &str) -> std::result::Result<usize, String> {
    parse_bounded_usize(value, MAXIMUM_INPUT_BYTES, "--maximum-input-bytes")
}

fn parse_bounded_usize(
    value: &str,
    maximum: usize,
    option_name: &str,
) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{option_name} must be a positive integer"))?;
    if !(1..=maximum).contains(&parsed) {
        return Err(format!("{option_name} must be between 1 and {maximum}"));
    }
    Ok(parsed)
}

/// Bound allocation before accepting each untrusted JSON Lines record.
fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    maximum_line_bytes: usize,
) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::with_capacity(maximum_line_bytes.min(8_192));
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let chunk_length = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(chunk_length) > maximum_line_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JSON Lines record exceeds {maximum_line_bytes} bytes"),
            ));
        }
        let has_newline = available[chunk_length - 1] == b'\n';
        line.extend_from_slice(&available[..chunk_length]);
        reader.consume(chunk_length);
        if has_newline {
            return Ok(Some(line));
        }
    }
}
