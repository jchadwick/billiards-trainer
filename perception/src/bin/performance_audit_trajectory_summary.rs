//! Bounded streaming JSON Lines adapter for trajectory receipt summaries.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use billiards_perception::trajectory_summary::TrajectoryAccumulator;
use clap::Parser;
use serde_json::Value;

const DEFAULT_MAXIMUM_SAMPLES: usize = 10_000;
const MAXIMUM_SAMPLES: usize = 1_000_000;
const DEFAULT_MAXIMUM_RECORDS: usize = 200_000;
const MAXIMUM_RECORDS: usize = 5_000_000;
const DEFAULT_MAXIMUM_LINE_BYTES: usize = 1_048_576;
const MAXIMUM_LINE_BYTES: usize = 16_777_216;
const DEFAULT_MAXIMUM_INPUT_BYTES: usize = 134_217_728;
const MAXIMUM_INPUT_BYTES: usize = 1_073_741_824;
const DEFAULT_MAXIMUM_OUTPUT_BYTES: usize = 1_048_576;
const MAXIMUM_OUTPUT_BYTES: usize = 16_777_216;

#[derive(Debug, Parser)]
#[command(about = "Stream bounded trajectory receipts into a native audit summary")]
struct Arguments {
    /// Maximum retained metric samples and source identities.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_SAMPLES, value_parser = parse_maximum_samples)]
    maximum_samples: usize,

    /// Maximum JSON Lines records, including config and finish records.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_RECORDS, value_parser = parse_maximum_records)]
    maximum_records: usize,

    /// Maximum bytes accepted for one input line, including its newline.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_LINE_BYTES, value_parser = parse_maximum_line_bytes)]
    maximum_line_bytes: usize,

    /// Maximum total input bytes accepted over the process lifetime.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_INPUT_BYTES, value_parser = parse_maximum_input_bytes)]
    maximum_input_bytes: usize,

    /// Maximum serialized summary bytes written to stdout.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_OUTPUT_BYTES, value_parser = parse_maximum_output_bytes)]
    maximum_output_bytes: usize,
}

struct Finished {
    duration_seconds: f64,
    exporter_loss_count: Option<u64>,
    bridge_loss_count: Option<u64>,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match run(arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("trajectory summary failed: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn run(arguments: Arguments) -> Result<()> {
    validate_limits(&arguments)?;
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut line_number = 0_u64;
    let mut input_bytes = 0_usize;
    let mut records = 0_usize;
    let mut accumulator = None;
    let mut finished = None;

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
            bail!("empty JSON Lines record at line {line_number}");
        }
        records = records
            .checked_add(1)
            .context("JSON Lines record counter overflowed")?;
        if records > arguments.maximum_records {
            bail!(
                "input exceeds maximum record count {}",
                arguments.maximum_records
            );
        }
        let record: Value = serde_json::from_slice(&line)
            .with_context(|| format!("invalid JSON record at line {line_number}"))?;
        let object = record
            .as_object()
            .context("each JSON Lines record must be an object")?;
        let record_type = object
            .get("type")
            .and_then(Value::as_str)
            .context("each record must have a string type")?;

        if finished.is_some() {
            bail!("finish record must be the final JSON Lines record");
        }

        if accumulator.is_none() {
            if record_type != "config" {
                bail!("first record must be the config record");
            }
            let maximum_samples =
                required_positive_integer(object.get("maximum_samples"), "maximum_samples")?;
            if maximum_samples > arguments.maximum_samples {
                bail!(
                    "configured maximum_samples {maximum_samples} exceeds CLI limit {}",
                    arguments.maximum_samples
                );
            }
            accumulator = Some(
                TrajectoryAccumulator::new(maximum_samples)
                    .map_err(anyhow::Error::msg)
                    .context("invalid trajectory summary configuration")?,
            );
            continue;
        }

        match record_type {
            "trajectory_receipt" => {
                let event = object
                    .get("event")
                    .context("trajectory_receipt record must contain event")?;
                accumulator
                    .as_mut()
                    .expect("configuration is validated before receipt records")
                    .add_event(event)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| format!("invalid trajectory receipt at line {line_number}"))?;
            }
            "finish" => {
                finished = Some(Finished {
                    duration_seconds: required_finite_non_negative_number(
                        object.get("duration_seconds"),
                        "duration_seconds",
                    )?,
                    exporter_loss_count: optional_non_negative_integer(
                        object.get("exporter_loss_count"),
                        "exporter_loss_count",
                    )?,
                    bridge_loss_count: optional_non_negative_integer(
                        object.get("bridge_loss_count"),
                        "bridge_loss_count",
                    )?,
                });
            }
            _ => bail!("record type must be trajectory_receipt or finish after config"),
        }
    }

    let finished = finished.context("input ended before the required finish record")?;
    let summary = accumulator
        .as_ref()
        .context("input is missing the required config record")?
        .summary(
            finished.duration_seconds,
            finished.exporter_loss_count,
            finished.bridge_loss_count,
        )
        .map_err(anyhow::Error::msg)
        .context("could not build trajectory summary")?;
    let mut output = serde_json::to_vec(&summary).context("could not serialize summary")?;
    output.push(b'\n');
    if output.len() > arguments.maximum_output_bytes {
        bail!(
            "summary output exceeds maximum byte count {}",
            arguments.maximum_output_bytes
        );
    }
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    stdout
        .write_all(&output)
        .context("could not write trajectory summary")?;
    stdout
        .flush()
        .context("could not flush trajectory summary")?;
    Ok(())
}

fn validate_limits(arguments: &Arguments) -> Result<()> {
    validate_limit(
        arguments.maximum_samples,
        MAXIMUM_SAMPLES,
        "--maximum-samples",
    )?;
    validate_limit(
        arguments.maximum_records,
        MAXIMUM_RECORDS,
        "--maximum-records",
    )?;
    validate_limit(
        arguments.maximum_line_bytes,
        MAXIMUM_LINE_BYTES,
        "--maximum-line-bytes",
    )?;
    validate_limit(
        arguments.maximum_input_bytes,
        MAXIMUM_INPUT_BYTES,
        "--maximum-input-bytes",
    )?;
    validate_limit(
        arguments.maximum_output_bytes,
        MAXIMUM_OUTPUT_BYTES,
        "--maximum-output-bytes",
    )?;
    Ok(())
}

fn validate_limit(value: usize, maximum: usize, name: &str) -> Result<()> {
    if !(1..=maximum).contains(&value) {
        bail!("{name} must be between 1 and {maximum}");
    }
    Ok(())
}

fn parse_maximum_samples(value: &str) -> std::result::Result<usize, String> {
    parse_limit(value, MAXIMUM_SAMPLES, "--maximum-samples")
}

fn parse_maximum_records(value: &str) -> std::result::Result<usize, String> {
    parse_limit(value, MAXIMUM_RECORDS, "--maximum-records")
}

fn parse_maximum_line_bytes(value: &str) -> std::result::Result<usize, String> {
    parse_limit(value, MAXIMUM_LINE_BYTES, "--maximum-line-bytes")
}

fn parse_maximum_input_bytes(value: &str) -> std::result::Result<usize, String> {
    parse_limit(value, MAXIMUM_INPUT_BYTES, "--maximum-input-bytes")
}

fn parse_maximum_output_bytes(value: &str) -> std::result::Result<usize, String> {
    parse_limit(value, MAXIMUM_OUTPUT_BYTES, "--maximum-output-bytes")
}

fn parse_limit(value: &str, maximum: usize, name: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer"))?;
    if !(1..=maximum).contains(&parsed) {
        return Err(format!("{name} must be between 1 and {maximum}"));
    }
    Ok(parsed)
}

fn required_positive_integer(value: Option<&Value>, name: &str) -> Result<usize> {
    let value = value.and_then(Value::as_u64).context(format!(
        "{name} must be a positive integer (booleans are invalid)"
    ))?;
    let value = usize::try_from(value).context(format!("{name} is too large"))?;
    if value == 0 {
        bail!("{name} must be positive");
    }
    Ok(value)
}

fn required_finite_non_negative_number(value: Option<&Value>, name: &str) -> Result<f64> {
    let value = value
        .and_then(Value::as_number)
        .and_then(serde_json::Number::as_f64)
        .context(format!(
            "{name} must be a finite non-negative number (booleans are invalid)"
        ))?;
    if !value.is_finite() || value < 0.0 {
        bail!("{name} must be finite and non-negative");
    }
    Ok(value)
}

fn optional_non_negative_integer(value: Option<&Value>, name: &str) -> Result<Option<u64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value.as_u64().map(Some).ok_or_else(|| {
        anyhow::anyhow!("{name} must be a non-negative integer or null (booleans are invalid)")
    })
}

/// Read one line without allowing an unbounded allocation for a malicious peer.
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
