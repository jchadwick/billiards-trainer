//! Bounded summaries for trajectory receipts observed by an audit client.
//!
//! This module consumes receipts synchronously in memory. Callers that need to
//! cross a process boundary should enqueue compact receipt records into a
//! bounded asynchronous writer; pipe writes do not belong on the websocket
//! receive path.

use std::collections::{HashMap, VecDeque};

use serde_json::{json, Map, Number, Value};

const IDENTITY_COMPONENT_LIMIT: usize = 256;

/// Bounded trajectory delivery counters and freshness samples.
#[derive(Debug)]
pub struct TrajectoryAccumulator {
    messages: u64,
    incomplete_messages: u64,
    stream_sequence_gaps: u64,
    stream_sequence_duplicates: u64,
    stream_sequence_out_of_order: u64,
    unique_source_observations: u64,
    repeated_source_observations: u64,
    source_sequence_gaps: u64,
    source_out_of_order: u64,
    invalidated_messages: u64,
    last_stream_sequence: Option<u64>,
    source_high_water: IdentityLru,
    ages: MetricSamples,
    source_to_receipt: MetricSamples,
}

impl TrajectoryAccumulator {
    /// Create an accumulator with bounded percentile and identity retention.
    pub fn new(maximum_samples: usize) -> Result<Self, String> {
        if maximum_samples == 0 {
            return Err("maximum_samples must be positive".to_owned());
        }
        Ok(Self {
            messages: 0,
            incomplete_messages: 0,
            stream_sequence_gaps: 0,
            stream_sequence_duplicates: 0,
            stream_sequence_out_of_order: 0,
            unique_source_observations: 0,
            repeated_source_observations: 0,
            source_sequence_gaps: 0,
            source_out_of_order: 0,
            invalidated_messages: 0,
            last_stream_sequence: None,
            source_high_water: IdentityLru::new(maximum_samples),
            ages: MetricSamples::new(maximum_samples),
            source_to_receipt: MetricSamples::new(maximum_samples),
        })
    }

    /// Add one flattened `trajectory_from_message` event plus receipt metrics.
    ///
    /// Missing sequence/frame values count as incomplete, matching the Python
    /// collector. Present values with the wrong JSON type are rejected so a
    /// malformed protocol event cannot silently distort ordering statistics.
    pub fn add_event(&mut self, event: &Value) -> Result<(), String> {
        let event = event
            .as_object()
            .ok_or_else(|| "trajectory event must be a JSON object".to_owned())?;
        let stream_sequence = optional_u64(event, "stream_sequence")?;
        let source_frame_number = optional_u64(event, "source_frame_number")?;
        let source_publication_monotonic_ns =
            optional_u64(event, "source_publication_monotonic_ns")?;
        let source_timestamp = optional_finite_number(event, "source_timestamp")?;
        let wall_age_estimate =
            optional_finite_number(event, "source_publication_wall_age_estimate_ms")?;
        let fallback_age = optional_finite_number(event, "source_publication_age_ms")?;
        let source_to_receipt =
            optional_finite_number(event, "source_publication_to_client_receipt_ms")?;
        let clock_comparable = optional_bool(event, "clock_comparable")?.unwrap_or(false);
        let prediction_status = optional_string(event, "prediction_status")?;
        let spatial_valid = optional_bool(event, "spatial_valid")?;
        let identity = source_identity(event)?;

        // The monotonic publication timestamp is validated even though the
        // bridge-provided duration is what gets sampled. This prevents a bool
        // or floating-point timestamp from entering the protocol unnoticed.
        let _ = (source_publication_monotonic_ns, source_timestamp);
        increment(&mut self.messages, "trajectory message count")?;
        if stream_sequence.is_none() || source_frame_number.is_none() {
            increment(
                &mut self.incomplete_messages,
                "incomplete trajectory message count",
            )?;
        }

        if let Some(sequence) = stream_sequence {
            if let Some(previous) = self.last_stream_sequence {
                if sequence > previous {
                    let gap = sequence - previous - 1;
                    add_counter(
                        &mut self.stream_sequence_gaps,
                        gap,
                        "trajectory stream sequence gaps",
                    )?;
                    self.last_stream_sequence = Some(sequence);
                } else if sequence == previous {
                    increment(
                        &mut self.stream_sequence_duplicates,
                        "trajectory stream sequence duplicates",
                    )?;
                } else {
                    increment(
                        &mut self.stream_sequence_out_of_order,
                        "trajectory stream sequence out-of-order count",
                    )?;
                }
            } else {
                self.last_stream_sequence = Some(sequence);
            }
        }

        if let Some(source_frame) = source_frame_number {
            match self.source_high_water.observe(identity, source_frame)? {
                SourceObservation::New => increment(
                    &mut self.unique_source_observations,
                    "unique source observation count",
                )?,
                SourceObservation::Repeated => increment(
                    &mut self.repeated_source_observations,
                    "repeated source observation count",
                )?,
                SourceObservation::Advanced(gap) => {
                    increment(
                        &mut self.unique_source_observations,
                        "unique source observation count",
                    )?;
                    add_counter(
                        &mut self.source_sequence_gaps,
                        gap,
                        "trajectory source sequence gaps",
                    )?;
                }
                SourceObservation::OutOfOrder => increment(
                    &mut self.source_out_of_order,
                    "trajectory source out-of-order count",
                )?,
            }
        }

        if prediction_status == Some("invalidated") || spatial_valid == Some(false) {
            increment(
                &mut self.invalidated_messages,
                "invalidated trajectory message count",
            )?;
        }

        if let Some(age) = wall_age_estimate.or(fallback_age) {
            self.ages.push(age)?;
        }
        if clock_comparable {
            if let Some(latency) = source_to_receipt {
                self.source_to_receipt.push(latency)?;
            }
        }
        Ok(())
    }

    /// Build the stable legacy summary and disclose bounded sample loss.
    pub fn summary(
        &self,
        duration_seconds: f64,
        exporter_loss_count: Option<u64>,
        bridge_loss_count: Option<u64>,
    ) -> Result<Value, String> {
        if !duration_seconds.is_finite() || duration_seconds < 0.0 {
            return Err("duration_seconds must be finite and non-negative".to_owned());
        }
        let rate = if duration_seconds == 0.0 {
            0.0
        } else {
            self.messages as f64 / duration_seconds
        };
        if !rate.is_finite() {
            return Err("trajectory_messages_per_second is not finite".to_owned());
        }
        Ok(json!({
            "trajectory_messages_received": self.messages,
            "trajectory_messages_per_second": rate,
            "incomplete_trajectory_messages": self.incomplete_messages,
            "trajectory_sequence_gaps": self.stream_sequence_gaps,
            "trajectory_sequence_duplicates": self.stream_sequence_duplicates,
            "trajectory_sequence_out_of_order_messages": self.stream_sequence_out_of_order,
            "trajectory_unique_source_observations": self.unique_source_observations,
            "trajectory_repeated_source_observations": self.repeated_source_observations,
            "trajectory_source_sequence_gaps": self.source_sequence_gaps,
            "trajectory_source_out_of_order_messages": self.source_out_of_order,
            "trajectory_source_identity_deduplication_window": self.source_high_water.summary(),
            "trajectory_invalidated_or_spatially_invalid_messages": self.invalidated_messages,
            "trajectory_source_publication_wall_age_estimate_ms": self.ages.summary(),
            "trajectory_source_publication_to_client_receipt_ms": self.source_to_receipt.summary(),
            "trajectory_exporter_dropped_records": exporter_loss_count,
            "trajectory_bridge_dropped_records": bridge_loss_count,
        }))
    }
}

#[derive(Debug)]
struct MetricSamples {
    limit: usize,
    total_count: u64,
    values: VecDeque<f64>,
}

impl MetricSamples {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            total_count: 0,
            values: VecDeque::with_capacity(limit),
        }
    }

    fn push(&mut self, value: f64) -> Result<(), String> {
        if !value.is_finite() {
            return Err("metric sample must be finite".to_owned());
        }
        increment(&mut self.total_count, "metric sample count")?;
        if self.values.len() == self.limit {
            self.values.pop_front();
        }
        self.values.push_back(value);
        Ok(())
    }

    fn summary(&self) -> Value {
        let mut ordered = self.values.iter().copied().collect::<Vec<_>>();
        ordered.sort_by(f64::total_cmp);
        let retained_count = ordered.len();
        json!({
            // `count` and `sample_limit` are preserved for parity with the
            // Python report schema; count intentionally means retained samples.
            "count": retained_count,
            "sample_limit": self.limit,
            "p50": nearest_rank(&ordered, 0.50),
            "p95": nearest_rank(&ordered, 0.95),
            "p99": nearest_rank(&ordered, 0.99),
            "total_count": self.total_count,
            "retained_count": retained_count,
            "samples_truncated": self.total_count > retained_count as u64,
        })
    }
}

fn nearest_rank(ordered: &[f64], rank: f64) -> Option<f64> {
    if ordered.is_empty() {
        return None;
    }
    let index = ((rank * ordered.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(ordered.len() - 1);
    Some(ordered[index])
}

#[derive(Debug)]
struct IdentityLru {
    capacity: usize,
    entries: HashMap<String, usize>,
    slots: Vec<IdentitySlot>,
    head: Option<usize>,
    tail: Option<usize>,
    evictions: u64,
}

#[derive(Debug)]
struct IdentitySlot {
    identity: String,
    high_water: u64,
    previous: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceObservation {
    New,
    Repeated,
    Advanced(u64),
    OutOfOrder,
}

impl IdentityLru {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity.min(16_384)),
            slots: Vec::with_capacity(capacity.min(16_384)),
            head: None,
            tail: None,
            evictions: 0,
        }
    }

    fn observe(
        &mut self,
        identity: String,
        frame_number: u64,
    ) -> Result<SourceObservation, String> {
        if let Some(index) = self.entries.get(&identity).copied() {
            let previous = self.slots[index].high_water;
            self.move_to_tail(index);
            if frame_number == previous {
                return Ok(SourceObservation::Repeated);
            }
            if frame_number < previous {
                return Ok(SourceObservation::OutOfOrder);
            }
            let gap = frame_number - previous - 1;
            self.slots[index].high_water = frame_number;
            return Ok(SourceObservation::Advanced(gap));
        }

        let index = if self.slots.len() < self.capacity {
            let index = self.slots.len();
            self.slots.push(IdentitySlot {
                identity: identity.clone(),
                high_water: frame_number,
                previous: self.tail,
                next: None,
            });
            if let Some(tail) = self.tail {
                self.slots[tail].next = Some(index);
            } else {
                self.head = Some(index);
            }
            self.tail = Some(index);
            index
        } else {
            let index = self
                .head
                .ok_or_else(|| "identity window is unexpectedly empty".to_owned())?;
            let old_identity = self.slots[index].identity.clone();
            self.entries.remove(&old_identity);
            increment(&mut self.evictions, "identity window eviction count")?;
            self.unlink(index);
            self.slots[index] = IdentitySlot {
                identity: identity.clone(),
                high_water: frame_number,
                previous: self.tail,
                next: None,
            };
            if let Some(tail) = self.tail {
                self.slots[tail].next = Some(index);
            } else {
                self.head = Some(index);
            }
            self.tail = Some(index);
            index
        };
        self.entries.insert(identity, index);
        Ok(SourceObservation::New)
    }

    fn move_to_tail(&mut self, index: usize) {
        if self.tail == Some(index) {
            return;
        }
        self.unlink(index);
        self.slots[index].previous = self.tail;
        self.slots[index].next = None;
        if let Some(tail) = self.tail {
            self.slots[tail].next = Some(index);
        } else {
            self.head = Some(index);
        }
        self.tail = Some(index);
    }

    fn unlink(&mut self, index: usize) {
        let previous = self.slots[index].previous;
        let next = self.slots[index].next;
        if let Some(previous) = previous {
            self.slots[previous].next = next;
        } else {
            self.head = next;
        }
        if let Some(next) = next {
            self.slots[next].previous = previous;
        } else {
            self.tail = previous;
        }
    }

    fn summary(&self) -> Value {
        json!({
            "policy": "least-recently-seen source identities",
            "capacity": self.capacity,
            "retained": self.entries.len(),
            "evictions": self.evictions,
        })
    }
}

fn source_identity(event: &Map<String, Value>) -> Result<String, String> {
    let instance = identity_component(event.get("vision_instance_id"), "vision_instance_id")?;
    let process = identity_component(event.get("source_process_id"), "source_process_id")?;
    Ok(format!("{instance}|{process}"))
}

fn identity_component(value: Option<&Value>, field: &str) -> Result<String, String> {
    let Some(value) = value else {
        return Ok("unknown".to_owned());
    };
    let value = match value {
        Value::Null => "unknown".to_owned(),
        Value::String(value) if value.is_empty() => "unknown".to_owned(),
        Value::String(value) => value.clone(),
        Value::Bool(false) => "unknown".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Number(number) if number.as_f64() == Some(0.0) => "unknown".to_owned(),
        Value::Number(number) => number.to_string(),
        _ => {
            return Err(format!(
                "{field} must be a string, number, boolean, or null"
            ))
        }
    };
    if value.len() > IDENTITY_COMPONENT_LIMIT {
        return Err(format!(
            "{field} exceeds the {IDENTITY_COMPONENT_LIMIT}-byte identity component limit"
        ));
    }
    Ok(value)
}

fn optional_u64(object: &Map<String, Value>, field: &str) -> Result<Option<u64>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_u64()
        .map(Some)
        .ok_or_else(|| format!("{field} must be a non-negative integer (booleans are invalid)"))
}

fn optional_finite_number(object: &Map<String, Value>, field: &str) -> Result<Option<f64>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let number = value
        .as_number()
        .and_then(Number::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| format!("{field} must be a finite number (booleans are invalid)"))?;
    Ok(Some(number))
}

fn optional_bool(object: &Map<String, Value>, field: &str) -> Result<Option<bool>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| format!("{field} must be a boolean or null"))
}

fn optional_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<Option<&'a str>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_str()
        .map(Some)
        .ok_or_else(|| format!("{field} must be a string or null"))
}

fn increment(value: &mut u64, field: &str) -> Result<(), String> {
    add_counter(value, 1, field)
}

fn add_counter(value: &mut u64, amount: u64, field: &str) -> Result<(), String> {
    *value = value
        .checked_add(amount)
        .ok_or_else(|| format!("{field} overflowed"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::TrajectoryAccumulator;
    use serde_json::json;

    #[test]
    fn ordering_identity_invalidation_and_percentiles_match_legacy_semantics() {
        let mut accumulator = TrajectoryAccumulator::new(8).unwrap();
        for event in [
            json!({"stream_sequence": 10, "source_frame_number": 5, "vision_instance_id": "v", "source_process_id": 1, "source_publication_wall_age_estimate_ms": 9.0, "source_publication_to_client_receipt_ms": 7.0, "clock_comparable": true}),
            json!({"stream_sequence": 12, "source_frame_number": 8, "vision_instance_id": "v", "source_process_id": 1, "prediction_status": "invalidated", "source_publication_wall_age_estimate_ms": 11.0, "source_publication_to_client_receipt_ms": 5.0, "clock_comparable": true}),
            json!({"stream_sequence": 12, "source_frame_number": 8, "vision_instance_id": "v", "source_process_id": 1, "spatial_valid": false, "source_publication_to_client_receipt_ms": 1.0, "clock_comparable": false}),
            json!({"stream_sequence": 11, "source_frame_number": 7, "vision_instance_id": "v", "source_process_id": 1}),
            json!({"stream_sequence": 13, "source_frame_number": 1, "vision_instance_id": "v", "source_process_id": 2}),
            json!({"stream_sequence": null, "source_frame_number": null}),
        ] {
            accumulator.add_event(&event).unwrap();
        }
        let summary = accumulator.summary(2.0, Some(3), Some(4)).unwrap();
        assert_eq!(summary["trajectory_messages_received"], 6);
        assert_eq!(summary["trajectory_messages_per_second"], 3.0);
        assert_eq!(summary["incomplete_trajectory_messages"], 1);
        assert_eq!(summary["trajectory_sequence_gaps"], 1);
        assert_eq!(summary["trajectory_sequence_duplicates"], 1);
        assert_eq!(summary["trajectory_sequence_out_of_order_messages"], 1);
        assert_eq!(summary["trajectory_unique_source_observations"], 3);
        assert_eq!(summary["trajectory_repeated_source_observations"], 1);
        assert_eq!(summary["trajectory_source_sequence_gaps"], 2);
        assert_eq!(summary["trajectory_source_out_of_order_messages"], 1);
        assert_eq!(
            summary["trajectory_invalidated_or_spatially_invalid_messages"],
            2
        );
        assert_eq!(
            summary["trajectory_source_publication_wall_age_estimate_ms"]["p50"],
            9.0
        );
        assert_eq!(
            summary["trajectory_source_publication_to_client_receipt_ms"]["count"],
            2
        );
        assert_eq!(summary["trajectory_exporter_dropped_records"], 3);
        assert_eq!(summary["trajectory_bridge_dropped_records"], 4);
    }

    #[test]
    fn lru_capacity_and_sample_truncation_are_explicit() {
        let mut accumulator = TrajectoryAccumulator::new(2).unwrap();
        for (frame, age) in [(1, 1.0), (2, 2.0), (3, 3.0)] {
            accumulator
                .add_event(&json!({
                    "source_frame_number": frame,
                    "stream_sequence": frame,
                    "vision_instance_id": format!("v{frame}"),
                    "source_process_id": 1,
                    "source_publication_wall_age_estimate_ms": age,
                }))
                .unwrap();
        }
        let summary = accumulator.summary(0.0, None, None).unwrap();
        assert_eq!(summary["trajectory_messages_per_second"], 0.0);
        assert_eq!(
            summary["trajectory_source_identity_deduplication_window"]["retained"],
            2
        );
        assert_eq!(
            summary["trajectory_source_identity_deduplication_window"]["evictions"],
            1
        );
        let samples = &summary["trajectory_source_publication_wall_age_estimate_ms"];
        assert_eq!(samples["count"], 2);
        assert_eq!(samples["total_count"], 3);
        assert_eq!(samples["retained_count"], 2);
        assert_eq!(samples["samples_truncated"], true);
        assert_eq!(samples["p50"], 2.0);
        assert_eq!(samples["p95"], 3.0);
        assert_eq!(summary["trajectory_exporter_dropped_records"], Value::Null);
    }

    #[test]
    fn missing_and_falsy_identity_components_share_the_unknown_identity() {
        let mut accumulator = TrajectoryAccumulator::new(2).unwrap();
        accumulator
            .add_event(&json!({"source_frame_number": 7}))
            .unwrap();
        accumulator
            .add_event(&json!({
                "source_frame_number": 7,
                "vision_instance_id": "",
                "source_process_id": 0
            }))
            .unwrap();
        let summary = accumulator.summary(1.0, None, None).unwrap();
        assert_eq!(summary["trajectory_unique_source_observations"], 1);
        assert_eq!(summary["trajectory_repeated_source_observations"], 1);
        assert_eq!(
            summary["trajectory_source_identity_deduplication_window"]["retained"],
            1
        );
    }

    #[test]
    fn rejects_boolean_and_malformed_numeric_fields_and_duration() {
        let mut accumulator = TrajectoryAccumulator::new(2).unwrap();
        for event in [
            json!({"stream_sequence": true}),
            json!({"source_frame_number": false}),
            json!({"source_timestamp": true}),
            json!({"source_publication_monotonic_ns": true}),
            json!({"source_publication_to_client_receipt_ms": true}),
            json!({"source_publication_wall_age_estimate_ms": "5"}),
        ] {
            assert!(accumulator.add_event(&event).is_err());
        }
        assert!(accumulator.summary(f64::NAN, None, None).is_err());
    }

    use serde_json::Value;
}
