//! Correlation and summarization for same-host audit delivery latency.
//!
//! A hostname or wall clock does not establish that monotonic timestamps share
//! an origin. The caller must supply the read-only server clock identity for
//! instrumentation-off runs; this reducer then requires exactly one client
//! and one server clock domain with identical values before subtracting any
//! cross-process timestamps.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{json, Value};

/// Correlate state and trajectory publication, socket-send, and client-receipt
/// timestamps without mixing monotonic clock domains.
pub fn summarize(
    client_events: &[Value],
    server_events: &[Value],
    deadline_budget_ms: f64,
    server_clock_domain_id: Option<&str>,
    clock_name: &str,
) -> Value {
    let mut publications: HashMap<(Option<String>, i128), i128> = HashMap::new();
    for event in server_events {
        if matches!(
            event_type(event),
            Some("state_published" | "trajectory_published")
        ) {
            if let Some((domain, sequence)) = domain_sequence(event) {
                if let Some(timestamp) = integer(event.get("event_monotonic_ns")) {
                    publications.insert((domain, sequence), timestamp);
                }
            }
        }
    }

    let mut deliveries: HashMap<(Option<String>, String, i128), i128> = HashMap::new();
    for event in server_events {
        if event_type(event) != Some("websocket_delivery_complete") {
            continue;
        }
        let Some(domain) = event.get("domain").and_then(Value::as_str) else {
            continue;
        };
        if !matches!(domain, "state" | "trajectory") {
            continue;
        }
        let Some(sequence) = integer(event.get("stream_sequence")) else {
            continue;
        };
        let Some(timestamp) = integer(event.get("event_monotonic_ns")) else {
            continue;
        };
        let Some(client_id) = optional_identity(event.get("client_id")) else {
            continue;
        };
        deliveries.insert((client_id, domain.to_owned(), sequence), timestamp);
    }

    let client_clock_domains: HashSet<String> = client_events
        .iter()
        .filter_map(|event| nonempty_string(event.get("clock_domain_id")))
        .collect();
    let mut server_clock_domains: HashSet<String> = server_events
        .iter()
        .filter_map(|event| nonempty_string(event.get("clock_domain_id")))
        .collect();
    if server_events.is_empty() {
        if let Some(clock_domain) = server_clock_domain_id.filter(|value| !value.is_empty()) {
            server_clock_domains.insert(clock_domain.to_owned());
        }
    }
    let clocks_comparable = client_clock_domains.len() == 1
        && server_clock_domains.len() == 1
        && client_clock_domains == server_clock_domains;
    let shared_clock_domain = client_clock_domains.iter().next();

    let source_to_state_publication: Vec<f64> = server_events
        .iter()
        .filter(|event| event_type(event) == Some("state_published") && clocks_comparable)
        .filter_map(|event| {
            let published = integer(event.get("event_monotonic_ns"))?;
            let source = integer(event.get("source_publication_monotonic_ns"))?;
            (source > 0 && published >= source).then_some((published - source) as f64 / 1e6)
        })
        .collect();

    let source_to_client_receipt: Vec<f64> = client_events
        .iter()
        .filter(|event| event_type(event) == Some("state_received") && clocks_comparable)
        .filter_map(|event| finite_number(event.get("source_publication_to_client_receipt_ms")))
        .collect();

    let mut state_publication_to_receipt = Vec::new();
    let mut trajectory_publication_to_receipt = Vec::new();
    let mut socket_to_receipt = Vec::new();
    let mut trajectory_socket_to_receipt = Vec::new();
    let mut received_state_sequences = HashSet::new();
    for event in client_events {
        let kind = event_type(event);
        let domain = match kind {
            Some("state_received") => Some("state"),
            Some("trajectory_received") => Some("trajectory"),
            _ => None,
        };
        if kind == Some("state_received") {
            if let Some(sequence) = integer(event.get("stream_sequence")) {
                received_state_sequences.insert(sequence);
            }
        }
        let Some(domain) = domain else { continue };
        let Some(sequence) = integer(event.get("stream_sequence")) else {
            continue;
        };
        let Some(received_ns) = integer(event.get("receive_monotonic_ns")) else {
            continue;
        };

        let publication = publications.get(&(Some(domain.to_owned()), sequence));
        if clocks_comparable {
            if let Some(publication_ns) = publication.filter(|timestamp| received_ns >= **timestamp)
            {
                let latency = (received_ns - publication_ns) as f64 / 1e6;
                if domain == "state" {
                    state_publication_to_receipt.push(latency);
                } else {
                    trajectory_publication_to_receipt.push(latency);
                }
            }
        }

        if clocks_comparable {
            if let Some(client_id) = optional_identity(event.get("client_id")) {
                let sent = deliveries.get(&(client_id, domain.to_owned(), sequence));
                if let Some(sent_ns) = sent.filter(|timestamp| received_ns >= **timestamp) {
                    let latency = (received_ns - sent_ns) as f64 / 1e6;
                    socket_to_receipt.push(latency);
                    if domain == "trajectory" {
                        trajectory_socket_to_receipt.push(latency);
                    }
                }
            }
        }
    }

    let mut component_ages: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for event in server_events
        .iter()
        .filter(|event| event_type(event) == Some("state_published"))
    {
        let Some(sequence) = integer(event.get("stream_sequence")) else {
            continue;
        };
        if !received_state_sequences.contains(&sequence) {
            continue;
        }
        let Some(ages) = event
            .get("component_age_seconds")
            .and_then(Value::as_object)
        else {
            continue;
        };
        for (component, age) in ages {
            if let Some(age) = finite_number(Some(age)) {
                component_ages
                    .entry(component.clone())
                    .or_default()
                    .push(age * 1000.0);
            }
        }
    }

    let deadline_misses = |values: &[f64]| {
        json!({
            "sample_count": values.len(),
            "missed_deadlines": values.iter().filter(|value| **value > deadline_budget_ms).count(),
        })
    };
    json!({
        "clock": clock_name,
        "clock_domain": if clocks_comparable {
            format!("verified OS monotonic clock domain: {}", shared_clock_domain.expect("one shared clock domain"))
        } else {
            "not comparable: client and server lack one shared verified OS monotonic clock domain".to_owned()
        },
        "clocks_comparable": clocks_comparable,
        "deadline_budget_ms": deadline_budget_ms,
        "source_publication_to_state_ms": stats(&source_to_state_publication),
        "source_publication_to_client_receipt_ms": stats(&source_to_client_receipt),
        "state_publication_to_receipt_ms": stats(&state_publication_to_receipt),
        "deadline_misses": {
            "source_publication_to_state": deadline_misses(&source_to_state_publication),
            "source_publication_to_client_receipt": deadline_misses(&source_to_client_receipt),
            "state_publication_to_receipt": deadline_misses(&state_publication_to_receipt),
        },
        "socket_send_to_receipt_ms": stats(&socket_to_receipt),
        "trajectory_publication_to_receipt_ms": stats(&trajectory_publication_to_receipt),
        "trajectory_socket_send_to_receipt_ms": stats(&trajectory_socket_to_receipt),
        "component_observation_age_at_publication_ms": component_ages
            .into_iter()
            .map(|(component, values)| (component, stats(&values)))
            .collect::<BTreeMap<_, _>>(),
    })
}

fn event_type(event: &Value) -> Option<&str> {
    event.get("event_type").and_then(Value::as_str)
}

fn domain_sequence(event: &Value) -> Option<(Option<String>, i128)> {
    Some((
        optional_string(event.get("domain")),
        integer(event.get("stream_sequence"))?,
    ))
}

fn optional_string(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_owned)
}

/// Keep absent/null client IDs equivalent while rejecting malformed identities.
fn optional_identity(value: Option<&Value>) -> Option<Option<String>> {
    match value {
        None | Some(Value::Null) => Some(None),
        Some(Value::String(value)) => Some(Some(value.clone())),
        Some(_) => None,
    }
}

fn nonempty_string(value: Option<&Value>) -> Option<String> {
    optional_string(value).filter(|value| !value.is_empty())
}

fn integer(value: Option<&Value>) -> Option<i128> {
    let value = value?;
    if value.is_boolean() {
        return None;
    }
    value
        .as_i64()
        .map(i128::from)
        .or_else(|| value.as_u64().map(i128::from))
}

fn finite_number(value: Option<&Value>) -> Option<f64> {
    let value = value?;
    if value.is_boolean() {
        return None;
    }
    value.as_f64().filter(|number| number.is_finite())
}

fn stats(values: &[f64]) -> Value {
    let mut values = values.to_vec();
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    let percentile = |rank: f64| -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let index = ((rank * values.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1);
        Some(values[index])
    };
    json!({
        "count": values.len(),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::summarize;

    const DOMAIN: &str = "darwin:boot-a:mach_absolute_time()";
    const CLOCK: &str = "mach_absolute_time";

    #[test]
    fn correlates_state_and_trajectory_delivery_inside_one_clock_domain() {
        let client = [
            json!({"event_type":"state_received", "clock_domain_id":DOMAIN, "stream_sequence":4, "client_id":"client-a", "receive_monotonic_ns":1050000, "source_publication_to_client_receipt_ms":0.08}),
            json!({"event_type":"trajectory_received", "clock_domain_id":DOMAIN, "stream_sequence":5, "client_id":"client-a", "receive_monotonic_ns":2080000}),
        ];
        let server = [
            json!({"event_type":"state_published", "clock_domain_id":DOMAIN, "domain":"state", "stream_sequence":4, "event_monotonic_ns":1000000, "source_publication_monotonic_ns":970000, "component_age_seconds":{"balls":0.02}}),
            json!({"event_type":"trajectory_published", "clock_domain_id":DOMAIN, "domain":"trajectory", "stream_sequence":5, "event_monotonic_ns":2000000}),
            json!({"event_type":"websocket_delivery_complete", "clock_domain_id":DOMAIN, "domain":"state", "stream_sequence":4, "client_id":"client-a", "event_monotonic_ns":1030000}),
            json!({"event_type":"websocket_delivery_complete", "clock_domain_id":DOMAIN, "domain":"trajectory", "stream_sequence":5, "client_id":"client-a", "event_monotonic_ns":2030000}),
        ];
        let output = summarize(&client, &server, 0.04, None, CLOCK);
        assert_eq!(output["clocks_comparable"], true);
        assert_eq!(output["source_publication_to_state_ms"]["p95"], 0.03);
        assert_eq!(
            output["source_publication_to_client_receipt_ms"]["p95"],
            0.08
        );
        assert_eq!(output["state_publication_to_receipt_ms"]["p95"], 0.05);
        assert_eq!(
            output["deadline_misses"]["state_publication_to_receipt"]["missed_deadlines"],
            1
        );
        assert_eq!(output["socket_send_to_receipt_ms"]["p50"], 0.02);
        assert_eq!(output["trajectory_socket_send_to_receipt_ms"]["p95"], 0.05);
        assert_eq!(
            output["component_observation_age_at_publication_ms"]["balls"]["p95"],
            20.0
        );
    }

    #[test]
    fn requires_one_equal_clock_domain_and_uses_read_only_baseline_identity() {
        let client = [
            json!({"event_type":"state_received", "clock_domain_id":DOMAIN, "stream_sequence":1, "source_publication_to_client_receipt_ms":14.0}),
            json!({"event_type":"state_received", "clock_domain_id":"other", "stream_sequence":2}),
        ];
        let output = summarize(&client, &[], 50.0, Some(DOMAIN), CLOCK);
        assert_eq!(output["clocks_comparable"], false);
        assert_eq!(
            output["source_publication_to_client_receipt_ms"]["count"],
            0
        );

        let single_client = [client[0].clone()];
        let baseline = summarize(&single_client, &[], 50.0, Some(DOMAIN), CLOCK);
        assert_eq!(baseline["clocks_comparable"], true);
        assert_eq!(
            baseline["source_publication_to_client_receipt_ms"]["p95"],
            14.0
        );
        assert_eq!(baseline["source_publication_to_state_ms"]["count"], 0);
    }

    #[test]
    fn hostname_and_boolean_timestamps_never_establish_or_fake_latency() {
        let client = [json!({
            "event_type":"state_received", "host_id":"same-host", "clock_domain_id":DOMAIN,
            "stream_sequence":true, "receive_monotonic_ns":true,
            "source_publication_to_client_receipt_ms":true
        })];
        let server = [json!({
            "event_type":"state_published", "host_id":"same-host", "clock_domain_id":"darwin:boot-b:mach_absolute_time()",
            "domain":"state", "stream_sequence":true, "event_monotonic_ns":true,
            "source_publication_monotonic_ns":1
        })];
        let output = summarize(&client, &server, 50.0, None, CLOCK);
        assert_eq!(output["clocks_comparable"], false);
        assert_eq!(output["state_publication_to_receipt_ms"]["count"], 0);
        assert_eq!(
            output["source_publication_to_client_receipt_ms"]["count"],
            0
        );
    }

    #[test]
    fn malformed_client_identity_does_not_match_anonymous_delivery() {
        let client = [json!({
            "event_type":"state_received", "clock_domain_id":DOMAIN,
            "stream_sequence":1, "receive_monotonic_ns":1050000
        })];
        let server = [
            json!({
                "event_type":"state_published", "clock_domain_id":DOMAIN,
                "domain":"state", "stream_sequence":1,
                "event_monotonic_ns":1000000
            }),
            json!({
                "event_type":"websocket_delivery_complete", "clock_domain_id":DOMAIN,
                "domain":"state", "stream_sequence":1, "client_id":42,
                "event_monotonic_ns":1030000
            }),
        ];
        let output = summarize(&client, &server, 50.0, None, CLOCK);
        assert_eq!(output["state_publication_to_receipt_ms"]["count"], 1);
        assert_eq!(output["socket_send_to_receipt_ms"]["count"], 0);
    }
}
