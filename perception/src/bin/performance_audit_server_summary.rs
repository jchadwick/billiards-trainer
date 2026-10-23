//! Bounded JSON Lines adapter for native server-event audit summaries.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use billiards_perception::server_audit_summary::summarize;
use clap::Parser;
use serde_json::Value;

const DEFAULT_MAXIMUM_EVENTS: usize = 200_000;
const MAXIMUM_EVENTS: usize = 5_000_000;
const DEFAULT_MAXIMUM_LINE_BYTES: usize = 1_048_576;
const MAXIMUM_LINE_BYTES: usize = 16_777_216;
const DEFAULT_MAXIMUM_INPUT_BYTES: usize = 134_217_728;
const MAXIMUM_INPUT_BYTES: usize = 1_073_741_824;

#[derive(Debug, Parser)]
#[command(
    about = "Summarize retained server performance audit events from bounded JSON Lines input"
)]
struct Arguments {
    /// Maximum event objects accepted from stdin.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_EVENTS, value_parser = parse_maximum_events)]
    maximum_events: usize,

    /// Maximum bytes accepted for any input line, including its newline.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_LINE_BYTES, value_parser = parse_maximum_line_bytes)]
    maximum_line_bytes: usize,

    /// Maximum total bytes accepted from stdin.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_INPUT_BYTES, value_parser = parse_maximum_input_bytes)]
    maximum_input_bytes: usize,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match run(arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("server audit summary failed: {error:#}");
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
    let mut events = Vec::new();
    let mut line_number = 0_u64;
    let mut input_bytes = 0_usize;
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
        if events.len() >= arguments.maximum_events {
            bail!(
                "input exceeds maximum event count {}",
                arguments.maximum_events
            );
        }
        let event: Value = serde_json::from_slice(&line)
            .with_context(|| format!("invalid JSON event at line {line_number}"))?;
        if !event.is_object() {
            bail!("event at line {line_number} must be a JSON object");
        }
        events.push(event);
    }

    let summary = summarize(&events);
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
