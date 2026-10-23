//! Reducers for retained server-side performance audit events.
//!
//! This module intentionally summarizes only events supplied by the caller.
//! It does not infer work from missing or sampled-out records, choose clock
//! windows, or merge final exporter status. Those policy decisions remain with
//! the audit orchestrator so this reducer can be checked against the retained
//! Python reference using the same event fixtures.

use std::collections::{BTreeMap, HashSet};

use serde_json::{json, Map, Value};

const TRAJECTORY_TIMING_OUTCOMES: [&str; 4] = [
    "trajectory_prediction_discarded_stale",
    "trajectory_prediction_failed",
    "trajectory_prediction_published",
    "trajectory_publication_failed",
];

#[derive(Default)]
struct StageSamples {
    inclusive: Vec<f64>,
    exclusive: Vec<f64>,
}

/// Summarize recorded server events into the established audit report shape.
///
/// Boolean values are rejected for numeric measurements even though JSON's
/// Python reference implementation inherits `bool` from `int`. Audit timings
/// and resource counters must represent explicit numeric measurements.
pub fn summarize(events: &[Value]) -> Value {
    let processed_unique = unique_observations(events, "vision_frame_complete");
    let integration_unique = unique_observations(events, "integration_observation_processed");
    let published_unique = unique_observations(events, "state_published");
    let processed_total = count_type(events, "vision_frame_complete");
    let integration_total = count_type(events, "integration_observation_processed");
    let published_total = count_type(events, "state_published");

    let resource_cpu = values(events, "process_resource_sample", "cpu_percent", None);
    let process_cpu_time = values(events, "process_resource_sample", "cpu_time_seconds", None);
    let resource_memory = values(
        events,
        "process_resource_sample",
        "resident_memory_bytes",
        None,
    );

    let mut process_events: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event_type(event) == Some("process_resource_sample"))
    {
        let key = format!(
            "{}:{}",
            python_field_string(event, "host_id", "unknown"),
            python_field_string(event, "process_id", "unknown")
        );
        process_events.entry(key).or_default().push(event);
    }
    let mut resources_by_process = Map::new();
    for (process_key, samples) in process_events {
        let cpu = values_for_events(&samples, "cpu_percent");
        let cpu_time = values_for_events(&samples, "cpu_time_seconds");
        let memory = integer_values_for_events(&samples, "resident_memory_bytes");
        let timed_memory: Vec<(i128, i128)> = samples
            .iter()
            .filter_map(|event| {
                Some((
                    integer(event.get("event_monotonic_ns")?)?,
                    integer(event.get("resident_memory_bytes")?)?,
                ))
            })
            .collect();
        let memory_span_seconds = (timed_memory.len() > 1).then(|| {
            let first = timed_memory[0].0;
            let last = timed_memory[timed_memory.len() - 1].0;
            (last - first).max(0) as f64 / 1e9
        });
        let memory_change = (timed_memory.len() > 1)
            .then(|| timed_memory[timed_memory.len() - 1].1 - timed_memory[0].1);
        let exporter_status = samples
            .iter()
            .rev()
            .find_map(|event| {
                event
                    .get("audit_export_status")
                    .filter(|value| value.is_object())
            })
            .cloned();
        let cpu_time_first = cpu_time.first().copied();
        let cpu_time_last = cpu_time.last().copied();
        let memory_first = memory.first().copied();
        let memory_last = memory.last().copied();
        let memory_rate = match (memory_change, memory_span_seconds) {
            (Some(change), Some(span)) if span > 0.0 => Some(change as f64 / span),
            _ => None,
        };
        resources_by_process.insert(
            process_key,
            json!({
                "sample_count": samples.len(),
                "cpu_percent": stats(cpu),
                "cpu_time_seconds": {
                    "first": cpu_time_first,
                    "last": cpu_time_last,
                    "delta": if cpu_time.len() > 1 { Some(cpu_time[cpu_time.len() - 1] - cpu_time[0]) } else { None },
                },
                "resident_memory_bytes": {
                    "first": memory_first,
                    "last": memory_last,
                    "minimum": memory.iter().min(),
                    "maximum": memory.iter().max(),
                    "change_first_to_last": memory_change,
                    "observed_span_seconds": memory_span_seconds,
                    "change_bytes_per_second": memory_rate,
                },
                "event_export_status": exporter_status,
            }),
        );
    }

    let event_loop_lag = values(
        events,
        "api_event_loop_sample",
        "event_loop_lag_seconds",
        None,
    );
    let outstanding_tasks = values(events, "api_event_loop_sample", "outstanding_tasks", None);
    let mut component_status_counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event_type(event) == Some("vision_frame_complete"))
    {
        let Some(statuses) = event
            .get("component_observation_status")
            .and_then(Value::as_object)
        else {
            continue;
        };
        for (component, status) in statuses {
            *component_status_counts
                .entry(component.clone())
                .or_default()
                .entry(value_string(status))
                .or_default() += 1;
        }
    }

    let vision_queue_depths = values(
        events,
        "vision_queue_frame_dropped",
        "queue_depth_after_drop",
        None,
    );
    let processing_queue_depths = values(
        events,
        "vision_frame_complete",
        "processing_queue_depth",
        None,
    );
    let mut stages: BTreeMap<String, StageSamples> = BTreeMap::new();
    for event in events
        .iter()
        .filter(|event| event_type(event) == Some("vision_frame_complete"))
    {
        for (field, inclusive) in [
            ("inclusive_stages_seconds", true),
            ("exclusive_stages_seconds", false),
        ] {
            let Some(timings) = event.get(field).and_then(Value::as_object) else {
                continue;
            };
            for (stage, duration) in timings {
                let Some(duration) = finite_number(duration) else {
                    continue;
                };
                if duration < 0.0 {
                    continue;
                }
                let samples = stages.entry(stage.clone()).or_default();
                if inclusive {
                    samples.inclusive.push(duration);
                } else {
                    samples.exclusive.push(duration);
                }
            }
        }
    }
    let stage_timings: BTreeMap<String, Value> = stages
        .iter()
        .map(|(stage, samples)| {
            let mut metrics = Map::new();
            if !samples.inclusive.is_empty() || !samples.exclusive.is_empty() {
                metrics.insert("inclusive".to_owned(), stats(samples.inclusive.clone()));
                metrics.insert("exclusive".to_owned(), stats(samples.exclusive.clone()));
            }
            (stage.clone(), Value::Object(metrics))
        })
        .collect();
    let mut ranked_stages: Vec<(String, f64, Value, Value)> = stages
        .iter()
        .filter(|(_, samples)| !samples.exclusive.is_empty())
        .map(|(stage, samples)| {
            (
                stage.clone(),
                samples.exclusive.iter().sum(),
                stats(samples.exclusive.clone()),
                stats(samples.inclusive.clone()),
            )
        })
        .collect();
    ranked_stages.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let total_exclusive_seconds: f64 = ranked_stages.iter().map(|stage| stage.1).sum();
    let ranked_exclusive_stages: Vec<Value> = ranked_stages
        .into_iter()
        .map(|(stage, total, exclusive, inclusive)| {
            json!({
                "stage": stage,
                "total_exclusive_seconds": total,
                "exclusive": exclusive,
                "inclusive": inclusive,
                "share_of_observed_exclusive_stage_seconds": if total_exclusive_seconds > 0.0 { Some(total / total_exclusive_seconds) } else { None },
            })
        })
        .collect();

    let prediction_outcome_counts: BTreeMap<&str, usize> = TRAJECTORY_TIMING_OUTCOMES
        .iter()
        .map(|event_type| (*event_type, count_type(events, event_type)))
        .collect();
    let prediction_outcome_event_count: usize = prediction_outcome_counts.values().sum();
    let source_publication_to_state_ms: Vec<f64> = events
        .iter()
        .filter(|event| event_type(event) == Some("state_published"))
        .filter_map(|event| {
            let event_ns = integer(event.get("event_monotonic_ns")?)?;
            let source_ns = integer(event.get("source_publication_monotonic_ns")?)?;
            (source_ns > 0 && event_ns >= source_ns).then_some((event_ns - source_ns) as f64 / 1e6)
        })
        .collect();

    json!({
        "record_count_semantics": "Counts below are derived from events retained by the asynchronous audit exporter. High-frequency pipeline and publication events are sampled; export queue drops and export limits can also reduce these counts. Use the headless subscriber's state accumulator for delivered-state throughput and repeat counts.",
        "vision": {
            "recorded_frame_detail_events": processed_total,
            "recorded_unique_source_observations_in_frame_details": processed_unique,
            "recorded_duplicate_or_reused_frame_details": processed_total.saturating_sub(processed_unique),
            "recorded_component_observation_status_counts": component_status_counts,
            "service_time_seconds": stats(values(events, "vision_frame_complete", "service_time_seconds", None)),
            "queue_wait_seconds": stats(values(events, "vision_frame_complete", "queue_delay_seconds", None)),
            "shared_memory_read_copy_seconds": stats(values(events, "vision_frame_complete", "shared_memory_read_copy_seconds", None)),
            "stage_timings_seconds": stage_timings,
            "exclusive_stage_work_ranked": ranked_exclusive_stages,
            "recorded_source_frame_rejection_events": count_type(events, "vision_source_frame_rejected"),
            "recorded_source_restart_events": count_type(events, "vision_source_restart_detected"),
            "recorded_vision_queue_drop_events": count_type(events, "vision_queue_frame_dropped"),
            "recorded_vision_queue_depth_after_drop_max": max_option(vision_queue_depths),
            "sampled_processing_queue_depth": stats(processing_queue_depths.clone()),
            "sampled_processing_queue_depth_max": max_option(processing_queue_depths),
        },
        "integration": {
            "recorded_observation_detail_events": integration_total,
            "recorded_unique_source_observations_in_observation_details": integration_unique,
            "recorded_observation_rejection_events": count_type(events, "integration_observation_rejected"),
            "coordinate_conversion_seconds": stats(values(events, "integration_observation_processed", "coordinate_conversion_seconds", None)),
            "core_update_seconds": stats(values(events, "integration_observation_processed", "core_update_seconds", None)),
            "integration_lock_wait_seconds": stats(values(events, "integration_observation_processed", "integration_lock_wait_seconds", None)),
        },
        "state_delivery": {
            "recorded_state_publication_detail_events": published_total,
            "recorded_unique_source_observations_in_publication_details": published_unique,
            "recorded_repeated_source_observation_publication_details": published_total.saturating_sub(published_unique),
            "serialization_seconds": stats(values(events, "websocket_message_serialized", "serialization_seconds", Some("state"))),
            "source_publication_to_state_ms": stats(source_publication_to_state_ms),
            "outbound_queue_wait_seconds": stats(values(events, "websocket_delivery_complete", "queue_wait_seconds", Some("state"))),
            "outbound_queue_depth_before_enqueue": stats(values(events, "websocket_message_serialized", "queue_depth_before_enqueue", Some("state"))),
            "recorded_coalesced_update_events": count_type_domain(events, "websocket_update_coalesced", "state"),
            "recorded_dropped_update_events": count_type_domain(events, "websocket_update_dropped", "state"),
            "socket_send_seconds": stats(values(events, "websocket_delivery_complete", "socket_send_seconds", Some("state"))),
            "payload_bytes": stats(values(events, "websocket_delivery_complete", "payload_bytes", Some("state"))),
        },
        "video_capture": {
            "camera_read_seconds": stats(values(events, "video_frame_published", "camera_read_seconds", None)),
            "camera_frame_queue_retrieval_seconds": stats(values(events, "video_frame_published", "camera_frame_queue_retrieval_seconds", None)),
            "undistortion_seconds": stats(values(events, "video_frame_published", "undistortion_seconds", None)),
            "shared_memory_publication_seconds": stats(values(events, "video_frame_published", "shared_memory_publication_seconds", None)),
            "camera_capture_queue_frames_skipped": sum_or_zero(values(events, "video_frame_published", "camera_capture_queue_frames_skipped", None)),
            "camera_total_dropped_frames_last": max_option(values(events, "video_frame_published", "camera_total_dropped_frames", None)),
        },
        "trajectory": {
            "published_messages": count_type(events, "trajectory_published"),
            "requests_coalesced": count_type(events, "trajectory_request_coalesced"),
            "predictions_discarded_stale": count_type(events, "trajectory_prediction_discarded_stale"),
            "prediction_outcome_counts": prediction_outcome_counts,
            "prediction_outcome_event_count": prediction_outcome_event_count,
            "prediction_queue_wait_seconds": stats(values_for_types(events, &TRAJECTORY_TIMING_OUTCOMES, "prediction_queue_wait_seconds")),
            "calculator_lock_wait_seconds": stats(values_for_types(events, &TRAJECTORY_TIMING_OUTCOMES, "calculator_lock_wait_seconds")),
            "executor_wait_seconds": stats(values_for_types(events, &TRAJECTORY_TIMING_OUTCOMES, "executor_wait_seconds")),
            "physics_compute_seconds": stats(values_for_types(events, &TRAJECTORY_TIMING_OUTCOMES, "physics_compute_seconds")),
            "trajectory_publication_seconds": stats(values_for_types(events, &TRAJECTORY_TIMING_OUTCOMES, "trajectory_publication_seconds")),
        },
        "resources": {
            "cpu_percent": stats(resource_cpu),
            "process_cpu_time_seconds": {
                "count": process_cpu_time.len(),
                "first": process_cpu_time.first(),
                "last": process_cpu_time.last(),
                "delta": if process_cpu_time.len() > 1 { Some(process_cpu_time[process_cpu_time.len() - 1] - process_cpu_time[0]) } else { None },
            },
            "resident_memory_bytes": {
                "count": resource_memory.len(),
                "first": resource_memory.first(),
                "last": resource_memory.last(),
                "minimum": resource_memory.iter().copied().reduce(f64::min),
                "maximum": resource_memory.iter().copied().reduce(f64::max),
            },
            "event_loop_lag_seconds": stats(event_loop_lag),
            "outstanding_tasks": stats(outstanding_tasks),
            "by_process": resources_by_process,
            "accelerator_metrics": "unavailable unless separately exposed by detector backend",
            "allocation_counts": "unavailable; RSS is a process memory proxy",
        },
        "viewers": {
            "encoded_frames": count_type(events, "viewer_frame_encoded"),
            "jpeg_encode_seconds": stats(values(events, "viewer_frame_encoded", "jpeg_encode_seconds", Some("viewers"))),
            "frame_read_seconds": stats(values(events, "viewer_frame_encoded", "frame_read_seconds", Some("viewers"))),
            "send_backpressure_seconds": stats(values(events, "viewer_send_backpressure", "send_backpressure_seconds", Some("viewers"))),
        },
    })
}

fn event_type(event: &Value) -> Option<&str> {
    event.get("event_type").and_then(Value::as_str)
}

fn count_type(events: &[Value], kind: &str) -> usize {
    events
        .iter()
        .filter(|event| event_type(event) == Some(kind))
        .count()
}

fn count_type_domain(events: &[Value], kind: &str, domain: &str) -> usize {
    events
        .iter()
        .filter(|event| {
            event_type(event) == Some(kind)
                && event.get("domain").and_then(Value::as_str) == Some(domain)
        })
        .count()
}

fn finite_number(value: &Value) -> Option<f64> {
    if value.is_boolean() {
        return None;
    }
    value.as_f64().filter(|number| number.is_finite())
}

fn integer(value: &Value) -> Option<i128> {
    if value.is_boolean() {
        return None;
    }
    value
        .as_i64()
        .map(i128::from)
        .or_else(|| value.as_u64().map(i128::from))
}

fn values(events: &[Value], kind: &str, field: &str, domain: Option<&str>) -> Vec<f64> {
    events
        .iter()
        .filter(|event| {
            event_type(event) == Some(kind)
                && domain.is_none_or(|expected| {
                    event.get("domain").and_then(Value::as_str) == Some(expected)
                })
        })
        .filter_map(|event| finite_number(event.get(field)?))
        .collect()
}

fn values_for_events(events: &[&Value], field: &str) -> Vec<f64> {
    events
        .iter()
        .filter_map(|event| finite_number(event.get(field)?))
        .collect()
}

fn values_for_types(events: &[Value], types: &[&str], field: &str) -> Vec<f64> {
    events
        .iter()
        .filter(|event| event_type(event).is_some_and(|kind| types.contains(&kind)))
        .filter_map(|event| finite_number(event.get(field)?))
        .collect()
}

fn integer_values_for_events(events: &[&Value], field: &str) -> Vec<i128> {
    events
        .iter()
        .filter_map(|event| integer(event.get(field)?))
        .collect()
}

fn unique_observations(events: &[Value], kind: &str) -> usize {
    let mut identities = HashSet::new();
    for event in events
        .iter()
        .filter(|event| event_type(event) == Some(kind))
    {
        let Some(frame_number) = event.get("source_frame_number").and_then(integer) else {
            continue;
        };
        let process_id = if event.get("vision_process_id").is_some() {
            event.get("vision_process_id")
        } else {
            event.get("process_id")
        };
        identities.insert((
            event.get("vision_instance_id").cloned(),
            process_id.cloned(),
            event.get("source_process_id").cloned(),
            frame_number,
        ));
    }
    identities.len()
}

fn stats(mut samples: Vec<f64>) -> Value {
    samples.retain(|value| value.is_finite());
    samples.sort_by(f64::total_cmp);
    let percentile = |rank: f64| -> Option<f64> {
        if samples.is_empty() {
            return None;
        }
        let index = ((rank * samples.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(samples.len() - 1);
        Some(samples[index])
    };
    json!({
        "count": samples.len(),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": if samples.is_empty() { None } else { Some(samples.iter().sum::<f64>() / samples.len() as f64) },
    })
}

fn max_option(values: Vec<f64>) -> Option<f64> {
    values.into_iter().reduce(f64::max)
}

fn sum_or_zero(values: Vec<f64>) -> Value {
    if values.is_empty() {
        json!(0)
    } else {
        json!(values.iter().sum::<f64>())
    }
}

fn value_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        other => other.to_string(),
    }
}

fn python_field_string(event: &Value, field: &str, missing: &str) -> String {
    event
        .get(field)
        .map(value_string)
        .unwrap_or_else(|| missing.to_owned())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::summarize;

    #[test]
    fn ranks_exclusive_stage_work_and_preserves_empty_measurements() {
        let output = summarize(&[
            json!({
                "event_type": "vision_frame_complete",
                "source_frame_number": 1,
                "service_time_seconds": 0.2,
                "queue_delay_seconds": 0.01,
                "processing_queue_depth": 0,
                "inclusive_stages_seconds": {"ball_detection": 0.16, "yolo_inference": 0.13},
                "exclusive_stages_seconds": {"ball_detection": 0.03, "yolo_inference": 0.13}
            }),
            json!({
                "event_type": "vision_frame_complete",
                "source_frame_number": 2,
                "service_time_seconds": 0.3,
                "queue_delay_seconds": 0.02,
                "processing_queue_depth": 3,
                "inclusive_stages_seconds": {"ball_detection": 0.22, "yolo_inference": 0.18},
                "exclusive_stages_seconds": {"ball_detection": 0.04, "yolo_inference": 0.18}
            }),
            json!({"event_type": "integration_observation_processed", "integration_lock_wait_seconds": 0.0005}),
        ]);
        assert_eq!(output["vision"]["service_time_seconds"]["p95"], 0.3);
        assert_eq!(output["vision"]["queue_wait_seconds"]["mean"], 0.015);
        assert_eq!(
            output["vision"]["stage_timings_seconds"]["yolo_inference"]["exclusive"]["p95"],
            0.18
        );
        assert_eq!(
            output["vision"]["exclusive_stage_work_ranked"][0]["stage"],
            "yolo_inference"
        );
        assert_eq!(
            output["integration"]["integration_lock_wait_seconds"]["p95"],
            0.0005
        );
        assert_eq!(
            output["trajectory"]["prediction_queue_wait_seconds"]["count"],
            0
        );
    }

    #[test]
    fn includes_all_trajectory_outcomes_and_rejects_boolean_metrics() {
        let kinds = [
            "trajectory_prediction_published",
            "trajectory_prediction_failed",
            "trajectory_prediction_discarded_stale",
            "trajectory_publication_failed",
        ];
        let events: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(index, kind)| json!({
                "event_type": kind,
                "prediction_queue_wait_seconds": index as f64 + 0.1,
                "physics_compute_seconds": index as f64 + 0.2,
                "trajectory_publication_seconds": if index == 0 || index == 3 { Some(0.03) } else { None },
            }))
            .chain([json!({"event_type": "process_resource_sample", "cpu_percent": true})])
            .collect();
        let output = summarize(&events);
        assert_eq!(output["trajectory"]["prediction_outcome_event_count"], 4);
        assert_eq!(
            output["trajectory"]["prediction_queue_wait_seconds"]["count"],
            4
        );
        assert_eq!(
            output["trajectory"]["trajectory_publication_seconds"]["count"],
            2
        );
        assert_eq!(output["resources"]["cpu_percent"]["count"], 0);
    }

    #[test]
    fn summarizes_per_process_resource_growth_and_latest_export_status() {
        let status = json!({"dropped_events": 3, "sampled_out_events": 4});
        let output = summarize(&[
            json!({"event_type": "process_resource_sample", "host_id": "audit-host", "process_id": 101, "event_monotonic_ns": 1_000_000_000_i64, "cpu_percent": 12.0, "cpu_time_seconds": 1.0, "resident_memory_bytes": 1024}),
            json!({"event_type": "process_resource_sample", "host_id": "audit-host", "process_id": 101, "event_monotonic_ns": 2_000_000_000_i64, "cpu_percent": 14.0, "cpu_time_seconds": 1.5, "resident_memory_bytes": 2048, "audit_export_status": status}),
        ]);
        let process = &output["resources"]["by_process"]["audit-host:101"];
        assert_eq!(
            process["resident_memory_bytes"]["change_first_to_last"],
            1024
        );
        assert_eq!(
            process["resident_memory_bytes"]["observed_span_seconds"],
            1.0
        );
        assert_eq!(
            process["resident_memory_bytes"]["change_bytes_per_second"],
            1024.0
        );
        assert_eq!(process["event_export_status"]["dropped_events"], 3);
    }

    #[test]
    fn matches_the_retained_python_reducer_golden_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/server-audit-summary.json"))
                .unwrap();
        let events = fixture["events"].as_array().unwrap();
        assert_eq!(summarize(events), fixture["expected"]);
    }
}
