//! Bounded JSON document adapter for native audit-overhead comparisons.

use std::io::{self, Read, Write};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use billiards_perception::overhead_comparison::{summarize, ComparisonInput};
use clap::Parser;
use serde_json::Value;

const DEFAULT_MAXIMUM_INPUT_BYTES: usize = 134_217_728;
const MAXIMUM_INPUT_BYTES: usize = 1_073_741_824;
const MAXIMUM_REPETITIONS: usize = 200_000;
const MAXIMUM_MINIMUM_REPETITIONS: usize = 10_000;
const MAXIMUM_DELIVERY_SAMPLES: usize = 10_000_000;

#[derive(Debug, Parser)]
#[command(about = "Compare bounded instrumentation-on and instrumentation-off audit reports")]
struct Arguments {
    /// Maximum bytes accepted from stdin.
    #[arg(long, default_value_t = DEFAULT_MAXIMUM_INPUT_BYTES, value_parser = parse_maximum_input_bytes)]
    maximum_input_bytes: usize,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match run(arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("audit overhead summary failed: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn run(arguments: Arguments) -> Result<()> {
    if !(1..=MAXIMUM_INPUT_BYTES).contains(&arguments.maximum_input_bytes) {
        bail!("--maximum-input-bytes must be between 1 and {MAXIMUM_INPUT_BYTES}");
    }
    let stdin = io::stdin();
    let mut bytes = Vec::with_capacity(arguments.maximum_input_bytes.min(65_536));
    stdin
        .lock()
        .take(arguments.maximum_input_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .context("could not read overhead comparison input")?;
    if bytes.len() > arguments.maximum_input_bytes {
        bail!(
            "input exceeds maximum byte count {}",
            arguments.maximum_input_bytes
        );
    }
    let document: Value = serde_json::from_slice(&bytes).context("invalid JSON input")?;
    let document = document
        .as_object()
        .context("overhead comparison input must be an object")?;
    let baseline_summary = document
        .get("baseline_summary")
        .and_then(Value::as_str)
        .context("baseline_summary must be a string")?;
    let current_metadata = required_object(document.get("current_metadata"), "current_metadata")?;
    let baseline_metadata =
        required_object(document.get("baseline_metadata"), "baseline_metadata")?;
    let current_repetitions =
        required_array(document.get("current_repetitions"), "current_repetitions")?;
    let baseline_repetitions =
        required_array(document.get("baseline_repetitions"), "baseline_repetitions")?;
    if current_repetitions.len() > MAXIMUM_REPETITIONS
        || baseline_repetitions.len() > MAXIMUM_REPETITIONS
    {
        bail!("each repetitions array must contain at most {MAXIMUM_REPETITIONS} runs");
    }
    if current_repetitions
        .iter()
        .chain(baseline_repetitions.iter())
        .any(|run| !run.is_object())
    {
        bail!("every repetition must be an object");
    }
    let budget_percent = document
        .get("budget_percent")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .context("budget_percent must be finite and non-negative")?;
    let minimum_repetitions = positive_limit(
        document.get("minimum_repetitions"),
        MAXIMUM_MINIMUM_REPETITIONS,
        "minimum_repetitions",
    )?;
    let minimum_delivery_samples = positive_limit(
        document.get("minimum_delivery_samples"),
        MAXIMUM_DELIVERY_SAMPLES,
        "minimum_delivery_samples",
    )?;

    let summary = summarize(&ComparisonInput {
        baseline_summary,
        current_metadata,
        current_repetitions,
        baseline_metadata,
        baseline_repetitions,
        budget_percent,
        minimum_repetitions,
        minimum_delivery_samples,
    });
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &summary).context("could not serialize summary")?;
    output
        .write_all(b"\n")
        .context("could not finish summary output")?;
    output.flush().context("could not flush summary output")?;
    Ok(())
}

fn required_object<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a Value> {
    value
        .filter(|value| value.is_object())
        .with_context(|| format!("{field} must be an object"))
}

fn required_array<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a [Value]> {
    value
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .with_context(|| format!("{field} must be an array"))
}

fn positive_limit(value: Option<&Value>, maximum: usize, field: &str) -> Result<usize> {
    let parsed = value
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .filter(|number| (1..=maximum).contains(number))
        .with_context(|| format!("{field} must be between 1 and {maximum}"))?;
    Ok(parsed)
}

fn parse_maximum_input_bytes(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| "--maximum-input-bytes must be a positive integer".to_owned())?;
    if !(1..=MAXIMUM_INPUT_BYTES).contains(&parsed) {
        return Err(format!(
            "--maximum-input-bytes must be between 1 and {MAXIMUM_INPUT_BYTES}"
        ));
    }
    Ok(parsed)
}
