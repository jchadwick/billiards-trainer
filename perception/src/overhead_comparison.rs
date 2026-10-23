//! Compare instrumentation-off and instrumentation-on audit repetitions.
//!
//! The Python collector owns baseline-file discovery and orchestration. This
//! reducer receives the two already-loaded reports and makes the comparison
//! decision without relying on host-local state or a wall clock.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

const COMPARABLE_CONFIGURATION_KEYS: [&str; 8] = [
    "camera",
    "detection",
    "processing",
    "vision_performance",
    "video",
    "calibration",
    "websocket_delivery",
    "stream_quality",
];
const ACTIVATION_KEYS: [&str; 4] = [
    "enabled",
    "state_enabled",
    "trajectory_enabled",
    "viewers_enabled",
];

/// Fully selected inputs for one cross-run comparison.
#[derive(Clone, Copy)]
pub struct ComparisonInput<'a> {
    pub baseline_summary: &'a str,
    pub current_metadata: &'a Value,
    pub current_repetitions: &'a [Value],
    pub baseline_metadata: &'a Value,
    pub baseline_repetitions: &'a [Value],
    pub budget_percent: f64,
    pub minimum_repetitions: usize,
    pub minimum_delivery_samples: usize,
}

/// Compare current audit repetitions with an instrumentation-off baseline.
pub fn summarize(input: &ComparisonInput<'_>) -> Value {
    let ComparisonInput {
        baseline_summary,
        current_metadata,
        current_repetitions,
        baseline_metadata,
        baseline_repetitions,
        budget_percent,
        minimum_repetitions,
        minimum_delivery_samples,
    } = *input;
    let current_throughput = average(current_repetitions, "unique_observations_per_second");
    let baseline_throughput = average(baseline_repetitions, "unique_observations_per_second");
    let current_p95 = average_source_delivery_p95(current_repetitions);
    let baseline_p95 = average_source_delivery_p95(baseline_repetitions);
    let throughput_change = percent_change(current_throughput, baseline_throughput);
    let p95_change = percent_change(current_p95, baseline_p95);

    let current_configuration = object_at(current_metadata, "effective_configuration");
    let baseline_configuration = object_at(baseline_metadata, "effective_configuration");
    let same_configuration = COMPARABLE_CONFIGURATION_KEYS
        .iter()
        .all(|key| equivalent_field(current_configuration, baseline_configuration, key));
    let same_audit_configuration =
        audit_configuration(current_configuration) == audit_configuration(baseline_configuration);

    let current_workloads = workload_signatures(current_repetitions);
    let baseline_workloads = workload_signatures(baseline_repetitions);
    let same_workload = current_workloads.len() == 1
        && baseline_workloads.len() == 1
        && current_workloads == baseline_workloads
        && current_workloads
            .values()
            .next()
            .is_some_and(|workload| workload.streams_are_nonempty);
    let reference_workload = same_workload
        && current_workloads.values().next().is_some_and(|workload| {
            workload.streams == vec![Value::String("state".to_owned())]
                && workload.video_url.is_null()
                && finite_number(Some(&workload.subscriber_delay_ms)) == Some(0.0)
        });

    let same_collector_settings = same_singleton_object_setting_set(current_repetitions)
        && same_singleton_object_setting_set(baseline_repetitions)
        && collector_settings(current_repetitions) == collector_settings(baseline_repetitions);
    let same_model = matching_nonempty_string(
        value_at(current_metadata, &["model", "sha256"]),
        value_at(baseline_metadata, &["model", "sha256"]),
    );
    let same_input = matching_nonempty_string(
        value_at(current_metadata, &["configured_video_input", "sha256"]),
        value_at(baseline_metadata, &["configured_video_input", "sha256"]),
    );
    let same_revision = matching_nonempty_string(
        current_metadata.get("revision"),
        baseline_metadata.get("revision"),
    );
    let same_dirty_worktree = matching_nonempty_string(
        current_metadata.get("dirty_worktree_sha256"),
        baseline_metadata.get("dirty_worktree_sha256"),
    );
    let baseline_instrumentation_off = !baseline_repetitions.is_empty()
        && baseline_repetitions.iter().all(|run| {
            let Some(domains) = run.get("server_audit_domains").and_then(Value::as_object) else {
                return false;
            };
            domains.values().all(|value| !truthy(value))
        });
    let current_state_only = !current_repetitions.is_empty()
        && current_repetitions.iter().all(|run| {
            run.get("server_audit_domains")
                == Some(&json!({
                    "state": true,
                    "trajectory": false,
                    "viewers": false
                }))
        });
    let same_machine = ["host", "platform", "processor", "hardware"]
        .iter()
        .all(|key| equivalent_field(current_metadata, baseline_metadata, key));
    let enough_data = current_repetitions.len() >= minimum_repetitions
        && baseline_repetitions.len() >= minimum_repetitions
        && current_repetitions
            .iter()
            .chain(baseline_repetitions.iter())
            .all(|run| delivery_sample_count(run) >= minimum_delivery_samples as f64);
    let passes_budget = throughput_change.is_some_and(|change| change >= -budget_percent)
        && p95_change.is_some_and(|change| change <= budget_percent);
    let comparable = same_machine
        && baseline_instrumentation_off
        && current_state_only
        && same_revision
        && same_dirty_worktree
        && same_configuration
        && same_audit_configuration
        && same_workload
        && reference_workload
        && same_collector_settings
        && same_model
        && same_input;

    let status = if enough_data && comparable && passes_budget {
        "pass"
    } else if !enough_data
        || !comparable
        || current_throughput.is_none()
        || baseline_throughput.is_none()
        || current_p95.is_none()
        || baseline_p95.is_none()
    {
        "inconclusive"
    } else {
        "over_budget"
    };
    let interpretation = match status {
        "pass" => "Instrumentation overhead is within the configured budget.",
        "over_budget" => "Instrumentation overhead exceeds the configured budget.",
        _ => "No performance recommendation until longer or better-matched runs meet the acceptance budget.",
    };

    json!({
        "baseline_summary": baseline_summary,
        "same_revision": same_revision,
        "same_dirty_worktree": same_dirty_worktree,
        "same_model_hash": same_model,
        "same_input_hash": same_input,
        "same_relevant_configuration": same_configuration,
        "same_audit_configuration": same_audit_configuration,
        "same_collector_settings": same_collector_settings,
        "same_workload": same_workload,
        "reference_state_only_workload": reference_workload,
        "same_machine": same_machine,
        "baseline_instrumentation_off": baseline_instrumentation_off,
        "current_state_audit_only": current_state_only,
        "unique_observation_throughput_change_percent": throughput_change,
        "source_publication_to_client_receipt_p95_change_percent": p95_change,
        "overhead_budget_percent": budget_percent,
        "enough_repetitions_and_samples": enough_data,
        "status": status,
        "interpretation": interpretation,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkloadSignature {
    streams: Vec<Value>,
    streams_are_nonempty: bool,
    video_url: Value,
    subscriber_delay_ms: Value,
    warmup_seconds: Value,
    requested_measured_seconds: Value,
}

fn workload_signatures(runs: &[Value]) -> BTreeMap<String, WorkloadSignature> {
    runs.iter()
        .filter_map(|run| {
            let streams: Vec<Value> = match run.get("streams").and_then(Value::as_array) {
                Some(streams) => {
                    let mut streams = streams.clone();
                    streams.sort_by_key(|value| value.to_string());
                    streams
                }
                None => Vec::new(),
            };
            let streams_are_nonempty = !streams.is_empty();
            let video_url = run
                .get("video_viewer")
                .and_then(Value::as_object)
                .and_then(|video| video.get("url"))
                .cloned()
                .unwrap_or(Value::Null);
            let workload = WorkloadSignature {
                streams,
                streams_are_nonempty,
                video_url,
                subscriber_delay_ms: value_or_null(run.get("subscriber_delay_ms")),
                warmup_seconds: value_or_null(run.get("warmup_seconds")),
                requested_measured_seconds: value_or_null(run.get("requested_measured_seconds")),
            };
            let key = serde_json::to_string(&json!([
                &workload.streams,
                &workload.video_url,
                &workload.subscriber_delay_ms,
                &workload.warmup_seconds,
                &workload.requested_measured_seconds
            ]))
            .ok()?;
            Some((key, workload))
        })
        .collect()
}

fn average(runs: &[Value], field: &str) -> Option<f64> {
    let values: Vec<f64> = runs
        .iter()
        .filter_map(|run| finite_number(run.get(field)))
        .collect();
    average_values(&values)
}

fn average_source_delivery_p95(runs: &[Value]) -> Option<f64> {
    let values: Vec<f64> = runs
        .iter()
        .filter_map(|run| {
            run.get("source_publication_to_client_receipt_ms")
                .and_then(|summary| summary.get("p95"))
                .and_then(|value| finite_number(Some(value)))
        })
        .collect();
    average_values(&values)
}

fn average_values(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        let average = values.iter().sum::<f64>() / values.len() as f64;
        average.is_finite().then_some(average)
    }
}

fn percent_change(current: Option<f64>, baseline: Option<f64>) -> Option<f64> {
    let current = current?;
    let baseline = baseline?;
    if baseline == 0.0 {
        return None;
    }
    let change = (current - baseline) / baseline * 100.0;
    change.is_finite().then_some(change)
}

fn finite_number(value: Option<&Value>) -> Option<f64> {
    let value = value?;
    if value.is_boolean() {
        return None;
    }
    value.as_f64().filter(|number| number.is_finite())
}

fn audit_configuration(configuration: &Value) -> Map<String, Value> {
    configuration
        .get("performance_audit")
        .and_then(Value::as_object)
        .map(|audit| {
            audit
                .iter()
                .filter(|(key, _)| !ACTIVATION_KEYS.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

fn collector_settings(runs: &[Value]) -> BTreeSet<String> {
    runs.iter()
        .filter_map(|run| run.get("collector_settings"))
        .filter(|settings| settings.is_object())
        .filter_map(|settings| serde_json::to_string(settings).ok())
        .collect()
}

fn same_singleton_object_setting_set(runs: &[Value]) -> bool {
    runs.iter()
        .filter_map(|run| run.get("collector_settings"))
        .filter(|settings| settings.is_object())
        .count()
        >= 1
        && collector_settings(runs).len() == 1
}

fn delivery_sample_count(run: &Value) -> f64 {
    run.get("source_publication_to_client_receipt_ms")
        .and_then(|summary| summary.get("count"))
        .and_then(|count| finite_number(Some(count)))
        .unwrap_or(0.0)
}

fn matching_nonempty_string(current: Option<&Value>, baseline: Option<&Value>) -> bool {
    match (
        current.and_then(Value::as_str),
        baseline.and_then(Value::as_str),
    ) {
        (Some(current), Some(baseline)) => !current.is_empty() && current == baseline,
        _ => false,
    }
}

fn object_at<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
}

fn equivalent_field(left: &Value, right: &Value, key: &str) -> bool {
    left.get(key).unwrap_or(&Value::Null) == right.get(key).unwrap_or(&Value::Null)
}

fn value_or_null(value: Option<&Value>) -> Value {
    value.cloned().unwrap_or(Value::Null)
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::{summarize, ComparisonInput};

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/overhead-comparison.json")).unwrap()
    }

    fn summarize_fixture(input: &Value) -> Value {
        summarize(&ComparisonInput {
            baseline_summary: input["baseline_summary"].as_str().unwrap(),
            current_metadata: &input["current_metadata"],
            current_repetitions: input["current_repetitions"].as_array().unwrap(),
            baseline_metadata: &input["baseline_metadata"],
            baseline_repetitions: input["baseline_repetitions"].as_array().unwrap(),
            budget_percent: input["budget_percent"].as_f64().unwrap(),
            minimum_repetitions: input["minimum_repetitions"].as_u64().unwrap() as usize,
            minimum_delivery_samples: input["minimum_delivery_samples"].as_u64().unwrap() as usize,
        })
    }

    #[test]
    fn matches_the_shared_overhead_comparison_fixture() {
        let fixture = fixture();
        assert_eq!(summarize_fixture(&fixture["input"]), fixture["expected"]);
    }

    #[test]
    fn over_budget_and_inconclusive_outcomes_have_consistent_interpretations() {
        let fixture = fixture();
        let mut input = fixture["input"].clone();
        for run in input["current_repetitions"].as_array_mut().unwrap() {
            run["unique_observations_per_second"] = json!(90.0);
            run["source_publication_to_client_receipt_ms"]["p95"] = json!(120.0);
        }
        let over_budget = summarize_fixture(&input);
        assert_eq!(over_budget["status"], "over_budget");
        assert_eq!(
            over_budget["interpretation"],
            "Instrumentation overhead exceeds the configured budget."
        );

        input["current_repetitions"][0]["unique_observations_per_second"] = json!(98.0);
        input["current_repetitions"][0]["source_publication_to_client_receipt_ms"]["p95"] =
            json!(102.0);
        input["current_repetitions"][0]["source_publication_to_client_receipt_ms"]["count"] =
            json!(99);
        let inconclusive = summarize_fixture(&input);
        assert_eq!(inconclusive["status"], "inconclusive");
        assert_eq!(
            inconclusive["interpretation"],
            "No performance recommendation until longer or better-matched runs meet the acceptance budget."
        );
    }

    #[test]
    fn boolean_performance_metrics_are_not_numeric_evidence() {
        let fixture = fixture();
        let mut input = fixture["input"].clone();
        for run in input["current_repetitions"].as_array_mut().unwrap() {
            run["unique_observations_per_second"] = json!(true);
            run["source_publication_to_client_receipt_ms"]["p95"] = json!(true);
        }
        let output = summarize_fixture(&input);
        assert_eq!(
            output["unique_observation_throughput_change_percent"],
            Value::Null
        );
        assert_eq!(
            output["source_publication_to_client_receipt_p95_change_percent"],
            Value::Null
        );
        assert_eq!(output["status"], "inconclusive");
    }
}
