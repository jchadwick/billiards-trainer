//! Bounded JSON Lines bridge for the Rust live timing summary accumulator.
//!
//! The collector process owns the monotonic clock. This bridge accepts its
//! explicit measurement boundaries and receiver timestamps, and deliberately
//! never samples a local clock that could silently change their meaning.

use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use billiards_perception::live_timing_accumulator::RustLiveTimingAccumulator;
use clap::Parser;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Map, Value};

const DEFAULT_MAXIMUM_SAMPLES: usize = 10_000;
const MAXIMUM_ALLOWED_SAMPLES: usize = 1_000_000;
const DEFAULT_MAXIMUM_LINE_BYTES: usize = 1_048_576;
const MAXIMUM_ALLOWED_LINE_BYTES: usize = 16_777_216;

#[derive(Debug, Parser)]
#[command(
    name = "billiards-performance-audit-timing-summary",
    about = "Summarize bounded Rust live timing records received over JSON Lines"
)]
struct Arguments {
    /// Maximum timing records retained for detailed summaries and diagnostics.
    #[arg(
        long,
        default_value_t = DEFAULT_MAXIMUM_SAMPLES,
        value_parser = parse_maximum_samples
    )]
    maximum_samples: usize,

    /// Maximum bytes in any one JSON Lines input record, including its newline.
    #[arg(
        long,
        default_value_t = DEFAULT_MAXIMUM_LINE_BYTES,
        value_parser = parse_maximum_line_bytes
    )]
    maximum_line_bytes: usize,
}

#[derive(Debug, Default)]
struct MeasurementOrder {
    started_at_ns: Option<u64>,
    ended_at_ns: Option<u64>,
}

/// Deserialize only the outer envelope as a map so duplicate top-level keys
/// are rejected instead of silently taking their last value.
struct StrictObject(Map<String, Value>);

impl<'de> Deserialize<'de> for StrictObject {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StrictObjectVisitor;

        impl<'de> Visitor<'de> for StrictObjectVisitor {
            type Value = StrictObject;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object with unique field names")
            }

            fn visit_map<A>(self, mut access: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut fields = Map::new();
                while let Some((key, value)) = access.next_entry::<String, Value>()? {
                    if fields.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate envelope field {key:?}"
                        )));
                    }
                }
                Ok(StrictObject(fields))
            }
        }

        deserializer.deserialize_map(StrictObjectVisitor)
    }
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match run(arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("timing summary bridge failed: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn run(arguments: Arguments) -> Result<()> {
    validate_limits(arguments.maximum_samples, arguments.maximum_line_bytes)?;
    let stdin = io::stdin();
    let result = process_jsonl(
        &mut stdin.lock(),
        arguments.maximum_samples,
        arguments.maximum_line_bytes,
    )?;

    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &result).context("could not serialize timing result")?;
    output
        .write_all(b"\n")
        .context("could not finish timing result")?;
    output.flush().context("could not flush timing result")?;
    Ok(())
}

fn validate_limits(maximum_samples: usize, maximum_line_bytes: usize) -> Result<()> {
    if !(1..=MAXIMUM_ALLOWED_SAMPLES).contains(&maximum_samples) {
        bail!("--maximum-samples must be between 1 and {MAXIMUM_ALLOWED_SAMPLES}");
    }
    if !(1..=MAXIMUM_ALLOWED_LINE_BYTES).contains(&maximum_line_bytes) {
        bail!("--maximum-line-bytes must be between 1 and {MAXIMUM_ALLOWED_LINE_BYTES}");
    }
    Ok(())
}

fn parse_maximum_samples(value: &str) -> std::result::Result<usize, String> {
    parse_bounded_usize(value, MAXIMUM_ALLOWED_SAMPLES, "--maximum-samples")
}

fn parse_maximum_line_bytes(value: &str) -> std::result::Result<usize, String> {
    parse_bounded_usize(value, MAXIMUM_ALLOWED_LINE_BYTES, "--maximum-line-bytes")
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

fn process_jsonl<R: BufRead>(
    reader: &mut R,
    maximum_samples: usize,
    maximum_line_bytes: usize,
) -> Result<Value> {
    validate_limits(maximum_samples, maximum_line_bytes)?;
    let mut accumulator = RustLiveTimingAccumulator::new(maximum_samples)?;
    let mut measurement = MeasurementOrder::default();
    let mut measured_state_trace = VecDeque::new();
    let mut locally_truncated_state_trace = false;
    let mut result = None;
    let mut line_number = 0_u64;

    while let Some(line) = read_bounded_line(reader, maximum_line_bytes).with_context(|| {
        format!(
            "could not read JSON Lines input at line {}",
            line_number + 1
        )
    })? {
        line_number = line_number.saturating_add(1);
        if result.is_some() {
            bail!("input contains a record after finish at line {line_number}");
        }
        let StrictObject(envelope) = serde_json::from_slice(&line)
            .with_context(|| format!("invalid JSON envelope at line {line_number}"))?;
        let kind = envelope
            .get("kind")
            .and_then(Value::as_str)
            .with_context(|| format!("line {line_number} is missing string field `kind`"))?;

        match kind {
            "record" => {
                validate_fields(&envelope, &["kind", "record"], &["kind", "record"])
                    .with_context(|| format!("invalid record envelope at line {line_number}"))?;
                let record = envelope
                    .get("record")
                    .filter(|record| record.is_object())
                    .context("`record` must be a JSON object")?;
                accumulator.add(record.clone());
            }
            "state" => {
                validate_fields(&envelope, &["kind", "state"], &["kind", "state"])
                    .with_context(|| format!("invalid state envelope at line {line_number}"))?;
                let state = envelope
                    .get("state")
                    .and_then(Value::as_object)
                    .context("`state` must be a JSON object")?;
                validate_fields(
                    state,
                    &[
                        "engine_instance_id",
                        "source_session_id",
                        "source_frame_id",
                        "source_transport_sequence",
                    ],
                    &[
                        "engine_instance_id",
                        "source_session_id",
                        "source_frame_id",
                        "source_transport_sequence",
                    ],
                )
                .context("invalid state identity")?;
                for field in ["engine_instance_id", "source_session_id", "source_frame_id"] {
                    if !state
                        .get(field)
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty())
                    {
                        bail!("state identity field `{field}` must be a nonempty string");
                    }
                }
                if !state
                    .get("source_transport_sequence")
                    .is_some_and(Value::is_u64)
                {
                    bail!("state identity field `source_transport_sequence` must be a nonnegative integer");
                }

                if measured_state_trace.len() == maximum_samples {
                    measured_state_trace.pop_front();
                    locally_truncated_state_trace = true;
                }
                measured_state_trace.push_back(Value::Object(state.clone()));
            }
            "begin_measurement" => {
                validate_fields(
                    &envelope,
                    &["kind", "monotonic_ns"],
                    &["kind", "monotonic_ns"],
                )
                .with_context(|| {
                    format!("invalid begin_measurement envelope at line {line_number}")
                })?;
                if measurement.started_at_ns.is_some() {
                    bail!("measurement may only begin once (line {line_number})");
                }
                let monotonic_ns = required_u64(&envelope, "monotonic_ns", line_number)?;
                measurement.started_at_ns = Some(monotonic_ns);
                accumulator.begin_measurement(monotonic_ns);
            }
            "end_measurement" => {
                validate_fields(
                    &envelope,
                    &["kind", "monotonic_ns"],
                    &["kind", "monotonic_ns"],
                )
                .with_context(|| {
                    format!("invalid end_measurement envelope at line {line_number}")
                })?;
                let Some(started_at_ns) = measurement.started_at_ns else {
                    bail!("measurement cannot end before it begins (line {line_number})");
                };
                if measurement.ended_at_ns.is_some() {
                    bail!("measurement may only end once (line {line_number})");
                }
                let monotonic_ns = required_u64(&envelope, "monotonic_ns", line_number)?;
                if monotonic_ns < started_at_ns {
                    bail!("measurement end precedes its start (line {line_number})");
                }
                measurement.ended_at_ns = Some(monotonic_ns);
                accumulator.end_measurement(monotonic_ns);
            }
            "finish" => {
                validate_fields(
                    &envelope,
                    &["kind", "state_trace_truncated", "bridge_dropped_records"],
                    &["kind", "state_trace_truncated"],
                )
                .with_context(|| format!("invalid finish envelope at line {line_number}"))?;
                if measurement.started_at_ns.is_some() && measurement.ended_at_ns.is_none() {
                    bail!("finish cannot close an unfinished measurement (line {line_number})");
                }
                let producer_state_trace_truncated = envelope
                    .get("state_trace_truncated")
                    .and_then(Value::as_bool)
                    .context("`state_trace_truncated` must be a boolean")?;
                let state_trace_truncated =
                    producer_state_trace_truncated || locally_truncated_state_trace;
                let bridge_dropped_records = envelope
                    .get("bridge_dropped_records")
                    .map(|value| {
                        value
                            .as_u64()
                            .context("`bridge_dropped_records` must be a nonnegative integer")
                    })
                    .transpose()?
                    .unwrap_or(0);

                let retained_state_trace: Vec<Value> =
                    measured_state_trace.iter().cloned().collect();
                let mut summary = accumulator.summary(&retained_state_trace, state_trace_truncated);
                if bridge_dropped_records > 0 {
                    force_partial_status(&mut summary);
                }
                result = Some(json!({
                    "summary": summary,
                    "stale_inference_timing_diagnostic": accumulator.stale_inference_diagnostic_summary(),
                    "bridge_dropped_records": bridge_dropped_records,
                }));
            }
            _ => bail!("unknown protocol kind {kind:?} at line {line_number}"),
        }
    }

    result.context("input ended before a finish envelope was received")
}

fn required_u64(envelope: &Map<String, Value>, field: &str, line_number: u64) -> Result<u64> {
    envelope
        .get(field)
        .and_then(Value::as_u64)
        .with_context(|| format!("`{field}` must be a nonnegative integer at line {line_number}"))
}

fn validate_fields(
    envelope: &Map<String, Value>,
    allowed: &[&str],
    required: &[&str],
) -> Result<()> {
    for field in envelope.keys() {
        if !allowed.contains(&field.as_str()) {
            bail!("unknown envelope field {field:?}");
        }
    }
    for field in required {
        if !envelope.contains_key(*field) {
            bail!("missing envelope field {field:?}");
        }
    }
    Ok(())
}

fn force_partial_status(summary: &mut Value) {
    if let Some(object) = summary.as_object_mut() {
        object.insert("status".to_owned(), json!("partial"));
        if let Some(pressure) = object
            .get_mut("association_capacity_pressure")
            .and_then(Value::as_object_mut)
        {
            pressure.insert("measurement_status".to_owned(), json!("partial"));
        }
    }
}

/// Read at most the configured limit into a line buffer. `read_until` and
/// `read_line` can grow their destination to match an untrusted line before a
/// caller gets a chance to check its length, so consume the `BufRead` chunks
/// incrementally and reject an overlong line before extending our buffer.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn run(input: &str, maximum_samples: usize, maximum_line_bytes: usize) -> Result<Value> {
        process_jsonl(
            &mut Cursor::new(input.as_bytes()),
            maximum_samples,
            maximum_line_bytes,
        )
    }

    fn error_message(result: Result<Value>) -> String {
        format!("{:#}", result.unwrap_err())
    }

    fn envelope(kind: &str, fields: &str) -> String {
        format!("{{\"kind\":\"{kind}\"{fields}}}\n")
    }

    fn frame(sequence: u64, receive_ns: u64, inference_ms: f64) -> Value {
        json!({
            "record_type": "frame",
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "frame_id": format!("frame-{sequence}"),
            "transport_sequence": sequence,
            "detection_count": 16,
            "track_count": 15,
            "durations_ms": { "inference": inference_ms },
            "audit_receiver_monotonic_ns": receive_ns,
        })
    }

    fn state(sequence: u64) -> Value {
        json!({
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "source_frame_id": format!("frame-{sequence}"),
            "source_transport_sequence": sequence,
        })
    }

    fn state_line(sequence: u64) -> String {
        format!("{}\n", json!({"kind":"state", "state":state(sequence)}))
    }

    fn record_line(record: Value) -> String {
        format!("{}\n", json!({"kind":"record", "record":record}))
    }

    fn finish_line(truncated: bool, dropped: Option<u64>) -> String {
        let mut value = json!({
            "kind": "finish",
            "state_trace_truncated": truncated,
        });
        if let Some(dropped) = dropped {
            value["bridge_dropped_records"] = json!(dropped);
        }
        format!("{value}\n")
    }

    fn complete_footer() -> Value {
        json!({
            "record_type": "footer",
            "export_status": "complete",
            "dropped_records": 0,
        })
    }

    #[test]
    fn includes_exact_measurement_boundaries_without_substituting_a_clock() {
        let input = format!(
            "{}{}{}{}{}{}{}{}{}",
            envelope("begin_measurement", ",\"monotonic_ns\":100"),
            record_line(frame(1, 99, 1.0)),
            record_line(frame(2, 100, 2.0)),
            record_line(frame(3, 200, 3.0)),
            record_line(frame(4, 201, 4.0)),
            envelope("end_measurement", ",\"monotonic_ns\":200"),
            state_line(2),
            state_line(3),
            finish_line(false, None),
        );
        let result = run(&input, 20, 8_192).unwrap();
        let summary = &result["summary"];
        assert_eq!(summary["measurement_monotonic_window_ns"]["start"], 100);
        assert_eq!(summary["measurement_monotonic_window_ns"]["end"], 200);
        assert_eq!(summary["stage_durations_ms"]["inference"]["count"], 2);
        assert_eq!(summary["stage_durations_ms"]["inference"]["p50"], 2.0);
        assert_eq!(summary["stage_durations_ms"]["inference"]["p99"], 3.0);
        assert_eq!(summary["state_timing_correlation"]["matched_frames"], 2);
        assert_eq!(
            summary["state_timing_correlation"]["timing_frames_without_measured_state"],
            0
        );
    }

    #[test]
    fn measurement_order_is_validated_and_finish_requires_a_closed_window() {
        let end_first = format!(
            "{}{}",
            envelope("end_measurement", ",\"monotonic_ns\":10"),
            finish_line(false, None)
        );
        assert!(error_message(run(&end_first, 10, 8_192)).contains("before it begins"));

        let backwards = format!(
            "{}{}{}",
            envelope("begin_measurement", ",\"monotonic_ns\":11"),
            envelope("end_measurement", ",\"monotonic_ns\":10"),
            finish_line(false, None)
        );
        assert!(error_message(run(&backwards, 10, 8_192)).contains("precedes its start"));

        let unfinished = format!(
            "{}{}",
            envelope("begin_measurement", ",\"monotonic_ns\":10"),
            finish_line(false, None)
        );
        assert!(error_message(run(&unfinished, 10, 8_192)).contains("unfinished measurement"));

        let duplicate_start = format!(
            "{}{}{}",
            envelope("begin_measurement", ",\"monotonic_ns\":10"),
            envelope("begin_measurement", ",\"monotonic_ns\":10"),
            finish_line(false, None)
        );
        assert!(error_message(run(&duplicate_start, 10, 8_192)).contains("begin once"));
    }

    #[test]
    fn finish_without_measurement_reports_startup_staleness_without_a_baseline() {
        let stale = json!({
            "record_type": "observation_outcome",
            "outcome": "stale",
            "reason": "stale_after_inference",
            "inference_timings_ms": {
                "inference": 8.5,
                "parent_record_read_complete_to_inference_complete": 24.0,
                "maximum_frame_age": 20.0,
            },
        });
        let input = format!("{}{}", record_line(stale), finish_line(false, None));
        let result = run(&input, 10, 8_192).unwrap();
        assert_eq!(result["summary"]["status"], "partial");
        assert_eq!(
            result["stale_inference_timing_diagnostic"]["stale_outcomes"],
            1
        );
        assert_eq!(
            result["stale_inference_timing_diagnostic"]["deadline_overrun_ms"]["p50"],
            4.0
        );
        assert_eq!(
            result["summary"]["measurement_monotonic_window_ns"]["start"],
            Value::Null
        );
    }

    #[test]
    fn bridge_drops_force_partial_even_when_the_producer_footer_is_complete() {
        let input = format!(
            "{}{}",
            record_line(complete_footer()),
            finish_line(false, Some(3))
        );
        let result = run(&input, 10, 8_192).unwrap();
        assert_eq!(result["bridge_dropped_records"], 3);
        assert_eq!(result["summary"]["status"], "partial");
        assert_eq!(
            result["summary"]["association_capacity_pressure"]["measurement_status"],
            "partial"
        );
    }

    #[test]
    fn exact_identity_join_does_not_match_a_different_session_or_sequence() {
        let mut other_session = frame(7, 5, 1.0);
        other_session["source_session_id"] = json!("session-b");
        let input = format!(
            "{}{}{}{}{}{}{}",
            envelope("begin_measurement", ",\"monotonic_ns\":5"),
            record_line(frame(7, 5, 2.0)),
            record_line(frame(8, 5, 99.0)),
            record_line(other_session),
            envelope("end_measurement", ",\"monotonic_ns\":5"),
            state_line(7),
            finish_line(false, None),
        );
        let result = run(&input, 10, 8_192).unwrap();
        assert_eq!(
            result["summary"]["state_timing_correlation"]["matched_frames"],
            1
        );
        assert_eq!(
            result["summary"]["state_timing_correlation"]["timing_frames_without_measured_state"],
            2
        );
        assert_eq!(
            result["summary"]["stage_durations_ms"]["inference"]["count"],
            1
        );
        assert_eq!(
            result["summary"]["stage_durations_ms"]["inference"]["p50"],
            2.0
        );
    }

    #[test]
    fn bounded_retention_preserves_whole_stream_counts_and_marks_truncation() {
        let input = format!(
            "{}{}{}{}{}{}{}",
            envelope("begin_measurement", ",\"monotonic_ns\":1"),
            record_line(frame(1, 1, 1.0)),
            record_line(frame(2, 1, 2.0)),
            state_line(1),
            state_line(2),
            envelope("end_measurement", ",\"monotonic_ns\":1"),
            finish_line(false, None),
        );
        let result = run(&input, 1, 8_192).unwrap();
        assert_eq!(result["summary"]["records_received"], 2);
        assert_eq!(result["summary"]["retained_records"], 1);
        assert_eq!(
            result["summary"]["state_timing_correlation"]["timing_trace_truncated"],
            true
        );
        assert_eq!(
            result["summary"]["state_timing_correlation"]["retained_measured_states"],
            1
        );
        assert_eq!(
            result["summary"]["state_timing_correlation"]["state_trace_truncated"],
            true
        );
    }

    #[test]
    fn rejects_unknown_missing_duplicate_and_malformed_envelope_fields() {
        let unknown = "{\"kind\":\"finish\",\"state_trace_truncated\":false,\"unexpected\":1}\n";
        assert!(error_message(run(unknown, 10, 8_192)).contains("unknown envelope field"));

        let missing = "{\"kind\":\"finish\"}\n";
        assert!(error_message(run(missing, 10, 8_192)).contains("missing envelope field"));

        let duplicate =
            "{\"kind\":\"finish\",\"kind\":\"finish\",\"state_trace_truncated\":false}\n";
        assert!(error_message(run(duplicate, 10, 8_192)).contains("duplicate envelope field"));

        let malformed = "{\"kind\":\"record\",\"record\":[]}\n";
        assert!(error_message(run(malformed, 10, 8_192)).contains("record` must be a JSON object"));

        let unknown_state_field = "{\"kind\":\"state\",\"state\":{\"engine_instance_id\":\"e\",\"source_session_id\":\"s\",\"source_frame_id\":\"f\",\"source_transport_sequence\":1,\"extra\":true}}\n";
        assert!(
            error_message(run(unknown_state_field, 10, 8_192)).contains("unknown envelope field")
        );

        let missing_state_field = "{\"kind\":\"state\",\"state\":{\"engine_instance_id\":\"e\",\"source_session_id\":\"s\",\"source_frame_id\":\"f\"}}\n";
        assert!(
            error_message(run(missing_state_field, 10, 8_192)).contains("missing envelope field")
        );
    }

    #[test]
    fn rejects_oversized_lines_and_data_after_finish() {
        let oversized = "{\"kind\":\"finish\",\"state_trace_truncated\":false}\n";
        assert!(error_message(run(oversized, 10, 16)).contains("exceeds 16 bytes"));

        let trailing = format!("{}{}", finish_line(false, None), "{{}}\n");
        assert!(error_message(run(&trailing, 10, 8_192)).contains("after finish"));
    }

    #[test]
    fn rejects_invalid_retention_and_protocol_configuration() {
        assert!(validate_limits(0, 10).is_err());
        assert!(validate_limits(MAXIMUM_ALLOWED_SAMPLES + 1, 10).is_err());
        assert!(validate_limits(1, 0).is_err());
        assert!(validate_limits(1, MAXIMUM_ALLOWED_LINE_BYTES + 1).is_err());

        let arguments = Arguments::try_parse_from(["timing-summary", "--maximum-samples", "0"]);
        assert!(arguments.is_err());
        let arguments =
            Arguments::try_parse_from(["timing-summary", "--maximum-line-bytes", "16777217"]);
        assert!(arguments.is_err());
    }
}
