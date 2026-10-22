//! Bounded aggregation of timing records emitted by the Rust live producer.
//!
//! The receive timestamp is intentionally compared only with the measurement
//! window supplied by the process that received the records. Producer clocks
//! and source clocks can belong to unrelated monotonic domains, so this module
//! never subtracts or otherwise compares them.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde_json::{json, Map, Number, Value};

const DURATION_FIELDS: &[&str] = &[
    "capture_helper_appsink_pull_wait",
    "capture_helper_rgb_buffer_allocation",
    "capture_helper_rgb_mapping_copy",
    "parent_framed_record_read",
    "parent_record_read_complete_to_inference_start",
    "parent_record_read_complete_to_scene_queue_submit",
    // Keep fields used by older, retained schema-v1 audit records.
    "host_read_complete_to_inference_start",
    "preprocess",
    "inference",
    "decode_and_nms",
    "engine_total",
    "calibration",
    "tracking_and_state_assembly",
    "scene_json_serialization",
    "scene_queue_submit",
    "host_read_complete_to_scene_queue_submit",
];

const INFERENCE_OUTCOME_DURATION_FIELDS: &[&str] = &[
    "capture_helper_appsink_pull_wait",
    "capture_helper_rgb_buffer_allocation",
    "capture_helper_rgb_mapping_copy",
    "parent_framed_record_read",
    "parent_record_read_complete_to_inference_start",
    "preprocess",
    "inference",
    "decode_and_nms",
    "engine_total",
    "parent_record_read_complete_to_inference_complete",
    "maximum_frame_age",
];

const PRESSURE_REASON_PREFIX: &str = "association_capacity_pressure_evicted_";

/// Failure returned when the accumulator is configured with no retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccumulatorError;

impl Display for AccumulatorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("maximum_samples must be positive")
    }
}

impl Error for AccumulatorError {}

/// Summarizes a bounded tail of producer events and tracks whole-run counters.
///
/// Detailed records are retained in a `VecDeque` with a fixed capacity. Counts
/// and the most recently seen footer cover the whole input stream, matching
/// the producer's footer semantics even if early records have been evicted.
#[derive(Debug, Clone)]
pub struct RustLiveTimingAccumulator {
    maximum_samples: usize,
    records: VecDeque<Value>,
    record_counts: BTreeMap<String, u64>,
    outcome_counts: BTreeMap<String, u64>,
    reason_counts: BTreeMap<String, u64>,
    measurement_started_ns: Option<u64>,
    measurement_ended_ns: Option<u64>,
    footer: Option<Value>,
    record_count: u64,
}

impl RustLiveTimingAccumulator {
    /// Create a bounded accumulator. A zero-sized retention buffer is invalid.
    pub fn new(maximum_samples: usize) -> Result<Self, AccumulatorError> {
        if maximum_samples == 0 {
            return Err(AccumulatorError);
        }
        Ok(Self {
            maximum_samples,
            // Do not reserve the full configured maximum up front: production
            // may use a large retention bound, but should allocate on demand.
            records: VecDeque::new(),
            record_counts: BTreeMap::new(),
            outcome_counts: BTreeMap::new(),
            reason_counts: BTreeMap::new(),
            measurement_started_ns: None,
            measurement_ended_ns: None,
            footer: None,
            record_count: 0,
        })
    }

    /// Set the local receive-clock start of the measured interval.
    pub fn begin_measurement(&mut self, monotonic_ns: u64) -> u64 {
        self.measurement_started_ns = Some(monotonic_ns);
        self.measurement_ended_ns = None;
        monotonic_ns
    }

    /// Set the local receive-clock end of the measured interval.
    pub fn end_measurement(&mut self, monotonic_ns: u64) -> u64 {
        self.measurement_ended_ns = Some(monotonic_ns);
        monotonic_ns
    }

    /// Retain one producer record while updating counters for every record.
    pub fn add(&mut self, record: Value) {
        self.record_count = self.record_count.saturating_add(1);
        let record_type = record.get("record_type").and_then(Value::as_str);
        if let Some(record_type) = record_type {
            increment(&mut self.record_counts, record_type);
        }
        if record_type == Some("observation_outcome") {
            if let Some(outcome) = record.get("outcome").and_then(Value::as_str) {
                increment(&mut self.outcome_counts, outcome);
            }
            if let Some(reason) = record.get("reason").and_then(Value::as_str) {
                increment(&mut self.reason_counts, reason);
            }
        }
        if record_type == Some("footer") {
            self.footer = Some(record.clone());
        }
        if self.records.len() == self.maximum_samples {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Return the retained diagnostic tail in arrival order.
    pub fn retained_records(&self) -> Vec<Value> {
        self.records.iter().cloned().collect()
    }

    /// Build the measured-window summary, joining only exact source identities.
    pub fn summary(&self, measured_state_trace: &[Value], state_trace_truncated: bool) -> Value {
        let state_keys: HashSet<Identity> = measured_state_trace
            .iter()
            .filter_map(|event| identity_key(event, false))
            .collect();

        let mut frame_records: HashMap<Identity, &Value> = HashMap::new();
        let mut duplicate_frame_records = 0_u64;
        let mut write_records_by_key: HashMap<Identity, Vec<&Value>> = HashMap::new();
        let mut write_attempts = 0_u64;
        let mut write_outcomes = BTreeMap::new();

        for record in &self.records {
            if !self.is_in_measurement_window(record) {
                continue;
            }
            let Some(key) = identity_key(record, true) else {
                continue;
            };
            match record.get("record_type").and_then(Value::as_str) {
                Some("frame") => match frame_records.entry(key) {
                    Entry::Vacant(entry) => {
                        entry.insert(record);
                    }
                    Entry::Occupied(_) => {
                        duplicate_frame_records = duplicate_frame_records.saturating_add(1);
                    }
                },
                Some("scene_socket_write") => {
                    write_attempts = write_attempts.saturating_add(1);
                    if let Some(outcome) = record.get("outcome").and_then(Value::as_str) {
                        increment(&mut write_outcomes, outcome);
                    }
                    write_records_by_key.entry(key).or_default().push(record);
                }
                _ => {}
            }
        }

        let frame_keys: HashSet<Identity> = frame_records.keys().cloned().collect();
        let matched_keys: HashSet<Identity> =
            state_keys.intersection(&frame_keys).cloned().collect();
        let unmatched_state_count = state_keys.difference(&matched_keys).count();
        let unmatched_timing_count = frame_records
            .keys()
            .filter(|key| !state_keys.contains(*key))
            .count();

        let mut durations = empty_duration_map(DURATION_FIELDS);
        let mut matched_detections = Vec::new();
        let mut matched_tracks = Vec::new();
        for key in &matched_keys {
            let Some(record) = frame_records.get(key) else {
                continue;
            };
            if let Some(raw_durations) = record.get("durations_ms").and_then(Value::as_object) {
                for field in DURATION_FIELDS {
                    if let Some(value) = raw_durations.get(*field).and_then(nonnegative_finite) {
                        durations
                            .get_mut(*field)
                            .expect("known duration field")
                            .push(value);
                    }
                }
            }
            if let Some(value) = record.get("detection_count").and_then(integer_as_f64) {
                matched_detections.push(value);
            }
            if let Some(value) = record.get("track_count").and_then(integer_as_f64) {
                matched_tracks.push(value);
            }
        }
        let stage_summary = duration_summaries(&durations, Some(self.maximum_samples));

        let matched_writes = matched_keys
            .iter()
            .flat_map(|key| write_records_by_key.get(key).into_iter().flatten().copied());
        let mut matched_write_durations: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        matched_write_durations.insert("written", Vec::new());
        matched_write_durations.insert("failed", Vec::new());
        for event in matched_writes {
            let Some(outcome) = event.get("outcome").and_then(Value::as_str) else {
                continue;
            };
            let Some(values) = matched_write_durations.get_mut(outcome) else {
                continue;
            };
            if let Some(duration) = event.get("duration_ms").and_then(nonnegative_finite) {
                values.push(duration);
            }
        }
        let write_duration_summary = Value::Object(
            matched_write_durations
                .into_iter()
                .map(|(outcome, values)| (outcome.to_owned(), percentile_object(&values, None)))
                .collect(),
        );

        let mut measured_outcomes = BTreeMap::new();
        let mut measured_stale_outcomes = 0_u64;
        let mut timed_stale_outcomes = 0_u64;
        let mut stale_durations = empty_duration_map(INFERENCE_OUTCOME_DURATION_FIELDS);
        let mut stale_deadline_overruns = Vec::new();
        let mut out_of_window_outcomes = 0_u64;
        let mut association_pressure_frames = 0_u64;
        let mut association_capacity_evictions = 0_u64;
        let mut malformed_pressure_outcomes = 0_u64;

        for record in &self.records {
            if record.get("record_type").and_then(Value::as_str) != Some("observation_outcome") {
                continue;
            }
            if self.is_in_measurement_window(record) {
                if let Some(outcome) = record.get("outcome").and_then(Value::as_str) {
                    increment(&mut measured_outcomes, outcome);
                }
                if record.get("reason").and_then(Value::as_str) == Some("stale_after_inference") {
                    measured_stale_outcomes = measured_stale_outcomes.saturating_add(1);
                    if let Some(raw_timings) = record
                        .get("inference_timings_ms")
                        .and_then(Value::as_object)
                    {
                        timed_stale_outcomes = timed_stale_outcomes.saturating_add(1);
                        collect_duration_values(raw_timings, &mut stale_durations);
                        if let (Some(age), Some(maximum_age)) = (
                            raw_timings
                                .get("parent_record_read_complete_to_inference_complete")
                                .and_then(nonnegative_finite),
                            raw_timings
                                .get("maximum_frame_age")
                                .and_then(nonnegative_finite),
                        ) {
                            stale_deadline_overruns.push((age - maximum_age).max(0.0));
                        }
                    }
                }
                if record.get("outcome").and_then(Value::as_str) == Some("degraded")
                    && record.get("stage").and_then(Value::as_str) == Some("track_association")
                {
                    let reason = record
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let suffix = reason
                        .strip_prefix(PRESSURE_REASON_PREFIX)
                        .unwrap_or_default();
                    if !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()) {
                        if let Ok(evictions) = suffix.parse::<u64>() {
                            association_pressure_frames =
                                association_pressure_frames.saturating_add(1);
                            association_capacity_evictions =
                                association_capacity_evictions.saturating_add(evictions);
                        } else {
                            malformed_pressure_outcomes =
                                malformed_pressure_outcomes.saturating_add(1);
                        }
                    } else {
                        malformed_pressure_outcomes = malformed_pressure_outcomes.saturating_add(1);
                    }
                }
            } else {
                out_of_window_outcomes = out_of_window_outcomes.saturating_add(1);
            }
        }

        let stale_inference_summary = json!({
            "stale_outcomes": measured_stale_outcomes,
            "timed_stale_outcomes": timed_stale_outcomes,
            "stage_durations_ms": duration_summaries(&stale_durations, Some(self.maximum_samples)),
            "deadline_overrun_ms": percentile_object(&stale_deadline_overruns, None),
        });

        let footer_pipeline_counters = self
            .footer
            .as_ref()
            .and_then(|footer| footer.get("pipeline_counters"))
            .and_then(Value::as_object);
        let footer_pressure_frames = footer_pipeline_counters
            .and_then(|counters| counters.get("association_capacity_pressure_frames"))
            .cloned()
            .unwrap_or(Value::Null);
        let footer_capacity_evictions = footer_pipeline_counters
            .and_then(|counters| counters.get("association_capacity_evictions"))
            .cloned()
            .unwrap_or(Value::Null);
        let producer_dropped_records = self
            .footer
            .as_ref()
            .and_then(|footer| footer.get("dropped_records"))
            .cloned();
        let footer_export_complete = self.footer.as_ref().is_some_and(|footer| {
            footer.get("export_status").and_then(Value::as_str) == Some("complete")
                && footer.get("dropped_records").is_none_or(is_numeric_zero)
        });
        let measurement_status = if footer_export_complete {
            "complete"
        } else {
            "partial"
        };

        json!({
            "status": measurement_status,
            "producer_dropped_records": producer_dropped_records,
            "record_counts": count_object(&self.record_counts),
            "observation_outcomes_received": count_object(&self.outcome_counts),
            "observation_outcomes_in_measurement_window": count_object(&measured_outcomes),
            "out_of_window_outcome_records": out_of_window_outcomes,
            "observation_outcome_reasons_received": count_object(&self.reason_counts),
            "association_capacity_pressure": {
                "measurement_status": measurement_status,
                "measured_window_pressure_frames": association_pressure_frames,
                "measured_window_evicted_tracks": association_capacity_evictions,
                "malformed_pressure_outcome_records": malformed_pressure_outcomes,
                "run_total_pressure_frames": footer_pressure_frames,
                "run_total_evicted_tracks": footer_capacity_evictions,
                "run_total_includes_warmup": true,
            },
            "state_timing_correlation": {
                "retained_measured_states": state_keys.len(),
                "retained_timing_frames": frame_records.len(),
                "matched_frames": matched_keys.len(),
                "measured_states_without_timing": unmatched_state_count,
                "timing_frames_without_measured_state": unmatched_timing_count,
                "duplicate_timing_frame_records": duplicate_frame_records,
                "state_trace_truncated": state_trace_truncated,
                "timing_trace_truncated": self.record_count > self.records.len() as u64,
                "identity_fields": [
                    "engine_instance_id",
                    "source_session_id",
                    "source_frame_id",
                    "transport_sequence",
                ],
            },
            "stage_durations_ms": stage_summary,
            "stale_inference_timing_ms": stale_inference_summary,
            "matched_detection_count": mean_summary(&matched_detections),
            "matched_track_count": mean_summary(&matched_tracks),
            "scene_socket_write_attempts_received": write_attempts,
            "scene_socket_write_outcomes_received": count_object(&write_outcomes),
            "matched_scene_socket_write_durations_ms": write_duration_summary,
            "measurement_monotonic_window_ns": {
                "start": self.measurement_started_ns,
                "end": self.measurement_ended_ns,
                "clock": "collector_process_perf_counter",
                "applies_to": "audit record receive times and outcome window only",
            },
            "footer": self.footer,
            "records_received": self.record_count,
            "retained_records": self.records.len(),
            "retention_limit": self.maximum_samples,
        })
    }

    /// Summarize retained startup stale results without claiming a baseline.
    pub fn stale_inference_diagnostic_summary(&self) -> Value {
        let stale_records: Vec<&Value> = self
            .records
            .iter()
            .filter(|record| {
                record.get("record_type").and_then(Value::as_str) == Some("observation_outcome")
                    && record.get("reason").and_then(Value::as_str) == Some("stale_after_inference")
            })
            .collect();
        let mut durations = empty_duration_map(INFERENCE_OUTCOME_DURATION_FIELDS);
        let mut deadline_overruns = Vec::new();
        let mut timed_records = 0_u64;
        for record in &stale_records {
            let Some(raw_timings) = record
                .get("inference_timings_ms")
                .and_then(Value::as_object)
            else {
                continue;
            };
            timed_records = timed_records.saturating_add(1);
            collect_duration_values(raw_timings, &mut durations);
            if let (Some(age), Some(maximum_age)) = (
                raw_timings
                    .get("parent_record_read_complete_to_inference_complete")
                    .and_then(nonnegative_finite),
                raw_timings
                    .get("maximum_frame_age")
                    .and_then(nonnegative_finite),
            ) {
                deadline_overruns.push((age - maximum_age).max(0.0));
            }
        }
        json!({
            "scope": "all_retained_run_records_not_a_measured_baseline",
            "records_received": self.record_count,
            "retained_records": self.records.len(),
            "retention_limit": self.maximum_samples,
            "records_truncated": self.record_count > self.records.len() as u64,
            "stale_outcomes": stale_records.len(),
            "timed_stale_outcomes": timed_records,
            "stage_durations_ms": duration_summaries(&durations, None),
            "deadline_overrun_ms": percentile_object(&deadline_overruns, None),
        })
    }

    fn is_in_measurement_window(&self, record: &Value) -> bool {
        let (Some(start), Some(end), Some(received)) = (
            self.measurement_started_ns,
            self.measurement_ended_ns,
            record
                .get("audit_receiver_monotonic_ns")
                .and_then(nonnegative_integer),
        ) else {
            return false;
        };
        start <= received && received <= end
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Identity {
    engine: String,
    session: String,
    frame: String,
    sequence: Sequence,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Sequence {
    Signed(i64),
    Unsigned(u64),
}

fn identity_key(event: &Value, timing: bool) -> Option<Identity> {
    let engine = nonempty_string(event.get("engine_instance_id")?)?;
    let session = nonempty_string(event.get("source_session_id")?)?;
    let frame_field = if timing {
        "frame_id"
    } else {
        "source_frame_id"
    };
    let sequence_field = if timing {
        "transport_sequence"
    } else {
        "source_transport_sequence"
    };
    let frame = nonempty_string(event.get(frame_field)?)?;
    let sequence = event.get(sequence_field)?;
    // `Value::as_u64` and `Value::as_i64` reject JSON booleans, as the source
    // implementation rejects Python bool despite bool being an int subclass.
    let sequence = sequence
        .as_i64()
        .map(Sequence::Signed)
        .or_else(|| sequence.as_u64().map(Sequence::Unsigned))?;
    Some(Identity {
        engine,
        session,
        frame,
        sequence,
    })
}

fn nonempty_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn increment(counts: &mut BTreeMap<String, u64>, key: &str) {
    let value = counts.entry(key.to_owned()).or_default();
    *value = value.saturating_add(1);
}

fn count_object(counts: &BTreeMap<String, u64>) -> Value {
    let mut object = Map::new();
    for (key, count) in counts {
        object.insert(key.clone(), json!(count));
    }
    Value::Object(object)
}

fn empty_duration_map(fields: &[&str]) -> BTreeMap<String, Vec<f64>> {
    fields
        .iter()
        .map(|field| ((*field).to_owned(), Vec::new()))
        .collect()
}

fn collect_duration_values(
    raw_values: &Map<String, Value>,
    durations: &mut BTreeMap<String, Vec<f64>>,
) {
    for (field, values) in durations.iter_mut() {
        if let Some(value) = raw_values.get(field).and_then(nonnegative_finite) {
            values.push(value);
        }
    }
}

fn duration_summaries(
    durations: &BTreeMap<String, Vec<f64>>,
    sample_limit: Option<usize>,
) -> Value {
    let mut summaries = Map::new();
    for (field, values) in durations {
        summaries.insert(field.clone(), percentile_object(values, sample_limit));
    }
    Value::Object(summaries)
}

fn percentile_object(values: &[f64], sample_limit: Option<usize>) -> Value {
    let mut ordered: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    ordered.sort_by(f64::total_cmp);
    let mut summary = Map::new();
    summary.insert("count".to_owned(), json!(ordered.len()));
    if let Some(sample_limit) = sample_limit {
        summary.insert("sample_limit".to_owned(), json!(sample_limit));
    }
    summary.insert(
        "p50".to_owned(),
        option_number(nearest_rank(&ordered, 0.50)),
    );
    summary.insert(
        "p95".to_owned(),
        option_number(nearest_rank(&ordered, 0.95)),
    );
    summary.insert(
        "p99".to_owned(),
        option_number(nearest_rank(&ordered, 0.99)),
    );
    Value::Object(summary)
}

/// Python collector parity: ceil(rank * n) - 1, clamped into the sample range.
fn nearest_rank(ordered: &[f64], rank: f64) -> Option<f64> {
    if ordered.is_empty() {
        return None;
    }
    let index = ((rank * ordered.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(ordered.len() - 1);
    ordered.get(index).copied()
}

fn option_number(value: Option<f64>) -> Value {
    value
        .and_then(Number::from_f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn nonnegative_finite(value: &Value) -> Option<f64> {
    let value = value.as_f64()?;
    (value.is_finite() && value >= 0.0).then_some(value)
}

fn nonnegative_integer(value: &Value) -> Option<u64> {
    value.as_u64()
}

fn is_numeric_zero(value: &Value) -> bool {
    match value {
        Value::Bool(value) => !value,
        Value::Number(value) => {
            value.as_i64() == Some(0) || value.as_u64() == Some(0) || value.as_f64() == Some(0.0)
        }
        _ => false,
    }
}

fn integer_as_f64(value: &Value) -> Option<f64> {
    value
        .as_i64()
        .map(|value| value as f64)
        .or_else(|| value.as_u64().map(|value| value as f64))
}

fn mean_summary(values: &[f64]) -> Value {
    json!({
        "count": values.len(),
        "mean": if values.is_empty() {
            Value::Null
        } else {
            option_number(Some(values.iter().sum::<f64>() / values.len() as f64))
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn matching_state(sequence: Value) -> Value {
        json!({
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "source_frame_id": "frame-0001",
            "source_transport_sequence": sequence,
        })
    }

    fn frame(sequence: Value, receive_ns: u64) -> Value {
        json!({
            "record_type": "frame",
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "frame_id": "frame-0001",
            "transport_sequence": sequence,
            "detection_count": 16,
            "track_count": 15,
            "durations_ms": { "inference": 2.4 },
            "audit_receiver_monotonic_ns": receive_ns,
        })
    }

    fn complete_footer(dropped: u64) -> Value {
        json!({
            "record_type": "footer",
            "export_status": "complete",
            "dropped_records": dropped,
            "pipeline_counters": {
                "association_capacity_pressure_frames": 1,
                "association_capacity_evictions": 2,
            }
        })
    }

    #[test]
    fn rejects_zero_retention_and_keeps_a_bounded_tail() {
        assert!(RustLiveTimingAccumulator::new(0).is_err());
        let mut accumulator = RustLiveTimingAccumulator::new(2).unwrap();
        accumulator.add(json!({ "record_type": "first" }));
        accumulator.add(json!({ "record_type": "second" }));
        accumulator.add(json!({ "record_type": "third" }));
        assert_eq!(
            accumulator.retained_records(),
            vec![
                json!({ "record_type": "second" }),
                json!({ "record_type": "third" })
            ]
        );
        let summary = accumulator.summary(&[], false);
        assert_eq!(summary["records_received"], 3);
        assert_eq!(summary["retained_records"], 2);
        assert_eq!(summary["retention_limit"], 2);
        assert_eq!(
            summary["state_timing_correlation"]["timing_trace_truncated"],
            true
        );
    }

    #[test]
    fn joins_only_exact_identities_inside_local_measurement_window() {
        let mut accumulator = RustLiveTimingAccumulator::new(8).unwrap();
        accumulator.begin_measurement(100);
        accumulator.add(frame(json!(7), 99)); // excluded by receive time
        accumulator.add(frame(json!(8), 150)); // same frame, wrong sequence
        accumulator.add(frame(json!(7), 151));
        accumulator.end_measurement(200);

        let summary = accumulator.summary(&[matching_state(json!(7))], false);
        assert_eq!(summary["state_timing_correlation"]["matched_frames"], 1);
        assert_eq!(
            summary["state_timing_correlation"]["timing_frames_without_measured_state"],
            1
        );
        assert_eq!(summary["stage_durations_ms"]["inference"]["p95"], 2.4);
        assert_eq!(summary["measurement_monotonic_window_ns"]["start"], 100);
        assert_eq!(summary["measurement_monotonic_window_ns"]["end"], 200);
    }

    #[test]
    fn mismatched_clock_domains_are_never_compared_or_subtracted() {
        let mut accumulator = RustLiveTimingAccumulator::new(4).unwrap();
        accumulator.begin_measurement(1_000);
        let mut record = frame(json!(1), 1_001);
        record["source_monotonic_ns"] = json!(u64::MAX);
        accumulator.add(record);
        accumulator.end_measurement(1_002);
        let summary = accumulator.summary(&[matching_state(json!(1))], false);
        assert_eq!(summary["state_timing_correlation"]["matched_frames"], 1);
    }

    #[test]
    fn reports_stale_inference_window_statistics_and_startup_diagnostics() {
        let mut accumulator = RustLiveTimingAccumulator::new(8).unwrap();
        accumulator.begin_measurement(10);
        accumulator.add(json!({
            "record_type": "observation_outcome",
            "outcome": "skipped",
            "reason": "stale_after_inference",
            "audit_receiver_monotonic_ns": 15,
            "inference_timings_ms": {
                "inference": 290.0,
                "engine_total": 330.0,
                "parent_record_read_complete_to_inference_complete": 365.0,
                "maximum_frame_age": 100.0,
            }
        }));
        accumulator.end_measurement(20);

        let measured = accumulator.summary(&[], false);
        let startup = accumulator.stale_inference_diagnostic_summary();
        assert_eq!(measured["stale_inference_timing_ms"]["stale_outcomes"], 1);
        assert_eq!(
            measured["stale_inference_timing_ms"]["timed_stale_outcomes"],
            1
        );
        assert_eq!(
            measured["stale_inference_timing_ms"]["stage_durations_ms"]["inference"]["p95"],
            290.0
        );
        assert_eq!(
            measured["stale_inference_timing_ms"]["deadline_overrun_ms"]["p95"],
            265.0
        );
        assert_eq!(
            startup["scope"],
            "all_retained_run_records_not_a_measured_baseline"
        );
        assert_eq!(startup["deadline_overrun_ms"]["p95"], 265.0);
    }

    #[test]
    fn tracks_footer_drop_status_socket_writes_and_pressure_suffixes() {
        let mut accumulator = RustLiveTimingAccumulator::new(16).unwrap();
        accumulator.begin_measurement(100);
        accumulator.add(frame(json!(7), 110));
        accumulator.add(json!({
            "record_type": "scene_socket_write",
            "engine_instance_id": "engine-a",
            "source_session_id": "session-a",
            "frame_id": "frame-0001",
            "transport_sequence": 7,
            "outcome": "written",
            "duration_ms": 0.45,
            "audit_receiver_monotonic_ns": 120,
        }));
        accumulator.add(json!({
            "record_type": "observation_outcome",
            "outcome": "degraded",
            "stage": "track_association",
            "reason": "association_capacity_pressure_evicted_2",
            "audit_receiver_monotonic_ns": 125,
        }));
        accumulator.add(json!({
            "record_type": "observation_outcome",
            "outcome": "degraded",
            "stage": "track_association",
            "reason": "association_capacity_pressure_evicted_two",
            "audit_receiver_monotonic_ns": 126,
        }));
        accumulator.add(complete_footer(0));
        accumulator.end_measurement(130);

        let summary = accumulator.summary(&[matching_state(json!(7))], false);
        assert_eq!(summary["status"], "complete");
        assert_eq!(summary["producer_dropped_records"], 0);
        assert_eq!(summary["scene_socket_write_attempts_received"], 1);
        assert_eq!(
            summary["matched_scene_socket_write_durations_ms"]["written"]["p95"],
            0.45
        );
        assert_eq!(
            summary["association_capacity_pressure"]["measured_window_pressure_frames"],
            1
        );
        assert_eq!(
            summary["association_capacity_pressure"]["measured_window_evicted_tracks"],
            2
        );
        assert_eq!(
            summary["association_capacity_pressure"]["malformed_pressure_outcome_records"],
            1
        );

        let mut dropped = RustLiveTimingAccumulator::new(2).unwrap();
        dropped.add(complete_footer(1));
        assert_eq!(dropped.summary(&[], false)["status"], "partial");
    }

    #[test]
    fn excludes_malformed_durations_and_rejects_boolean_identity_sequences() {
        let mut accumulator = RustLiveTimingAccumulator::new(8).unwrap();
        accumulator.begin_measurement(1);
        let mut record = frame(json!(7), 2);
        record["durations_ms"] = json!({
            "inference": 2.0,
            "preprocess": -1.0,
            "engine_total": "NaN",
            "calibration": true,
        });
        accumulator.add(record);
        accumulator.add(frame(json!(true), 3));
        accumulator.end_measurement(4);
        let summary = accumulator.summary(
            &[matching_state(json!(7)), matching_state(json!(true))],
            false,
        );
        assert_eq!(summary["state_timing_correlation"]["matched_frames"], 1);
        assert_eq!(summary["stage_durations_ms"]["inference"]["count"], 1);
        assert_eq!(summary["stage_durations_ms"]["preprocess"]["count"], 0);
        assert_eq!(summary["stage_durations_ms"]["engine_total"]["count"], 0);
        assert_eq!(summary["stage_durations_ms"]["calibration"]["count"], 0);
    }

    #[test]
    fn retains_legacy_stage_duration_aliases() {
        let mut accumulator = RustLiveTimingAccumulator::new(2).unwrap();
        accumulator.begin_measurement(10);
        let mut record = frame(json!(4), 11);
        record["durations_ms"] = json!({
            "host_read_complete_to_inference_start": 1.25,
            "host_read_complete_to_scene_queue_submit": 3.5,
        });
        accumulator.add(record);
        accumulator.end_measurement(12);
        let summary = accumulator.summary(&[matching_state(json!(4))], false);
        assert_eq!(
            summary["stage_durations_ms"]["host_read_complete_to_inference_start"]["p95"],
            1.25
        );
        assert_eq!(
            summary["stage_durations_ms"]["host_read_complete_to_scene_queue_submit"]["p95"],
            3.5
        );
    }

    #[test]
    fn distinguishes_missing_and_out_of_window_measurements() {
        let mut accumulator = RustLiveTimingAccumulator::new(4).unwrap();
        accumulator.add(json!({
            "record_type": "observation_outcome",
            "outcome": "skipped",
            "reason": "before_start",
            "audit_receiver_monotonic_ns": 2,
        }));
        accumulator.begin_measurement(3);
        accumulator.add(json!({
            "record_type": "observation_outcome",
            "outcome": "failed",
            "reason": "after_end",
            "audit_receiver_monotonic_ns": 5,
        }));
        accumulator.end_measurement(4);
        let summary = accumulator.summary(&[], false);
        assert_eq!(summary["out_of_window_outcome_records"], 2);
        assert!(summary["observation_outcomes_in_measurement_window"]
            .as_object()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn exact_integer_percentiles_use_nearest_rank() {
        assert_eq!(nearest_rank(&[1.0, 2.0, 3.0, 4.0], 0.50), Some(2.0));
        assert_eq!(nearest_rank(&[1.0, 2.0, 3.0, 4.0], 0.95), Some(4.0));
        assert_eq!(nearest_rank(&[], 0.95), None);
    }
}
