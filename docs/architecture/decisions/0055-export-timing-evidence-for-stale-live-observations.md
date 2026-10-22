# ADR-0055: Export timing evidence for stale live observations

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The target Linux webcam is visibly capturing the illuminated table and balls, but
the Rust live pipeline has discarded every completed inference as
`stale_after_inference`. The startup audit therefore has no usable localization
and no published state. The current audit emits only an identity and disposition
for skipped observations; stage timings are attached only to successful frame
records. As a result, the most consequential latency path cannot be diagnosed.

The detector already measures preprocessing, runtime inference, decode/NMS, and
total engine time. The capture envelope carries helper capture durations and a
parent-local monotonic timestamp from complete record read. The pipeline also
knows the interval to inference start, inference completion age, and configured
maximum frame age. The source does not provide a verified exposure timestamp,
and the helper and parent have independent monotonic clock origins. These
limits must remain explicit.

## Decision

Advance the opt-in Rust live-audit wire schema to version 3. Add an optional
`inference_timings_ms` object to observation outcomes when inference completed
but freshness validation rejected the result. It will carry supported helper
capture/read segments, parent-read-complete-to-inference-start, preprocessing,
runtime inference, decode/NMS, engine total, parent-read-complete-to-inference
completion, and the configured maximum frame age. Normal frame timing records
will also include engine total. The Python supervisor will strictly validate
the v3 field shape while continuing to accept archived v1 and v2 records; the
collector will summarize stale inference stages and age independently from
published-frame correlation.

Rejected stale observations keep their existing frame identity and revision
fields, null sequence where no scene was admitted, and nullable source
timestamp exactly as supplied. The audit must not invent camera exposure time,
publish a stale scene, or weaken the freshness guard. Timings continue to use
only process-local monotonic durations; no cross-process clock subtraction is
introduced.

## Consequences

### Positive

- A failed live readiness run can distinguish capture/read delay, inference
  scheduling, preprocessing, model execution, and post-inference freshness age.
- Stale frame diagnostics can be correlated by process, engine, source session,
  frame identity, and revisions without pretending a scene was published.
- Archived audit artifacts remain parseable, and normal state transport and
  latest-only freshness semantics remain unchanged.

### Negative

- Rust serialization, Python strict validation, and collector summaries must
  evolve together, with the schema version identifying the new contract.
- Optional audit records grow by a bounded set of scalar durations when a
  result is rejected after inference; instrumentation remains opt-in.
- Timings begin at parent framed-record completion, not camera exposure, and
  cannot account for the sensor's physical acquisition delay.

## Alternatives considered

### Keep stale outcomes timing-free

Rejected because the repeated target run produced no successful frame records,
so this leaves the exact failure mode unobservable and makes any performance
recommendation speculative.

### Disable the freshness limit or publish late results

Rejected because it would permit old states to replace current state and
contradict the accepted latest-state contract in ADR-0052. Instrumentation must
explain stale rejection without changing the freshness policy.

## Verification

- Rust serialization tests verify v3 frame engine-total and stale timing fields,
  including null source timestamps and no transport sequence.
- Rust pipeline tests verify a stale inference remains rejected and has no frame
  or scene publication record while retaining measured stage durations.
- Python strict schema tests accept v1/v2 compatibility and validate v3 timing
  values and malformed-field rejection; collector tests summarize stale timing
  percentiles only inside the measurement window.
- Run the Rust perception suite, the focused Python supervisor/collector tests,
  and the configured backend quality gate. Link target artifacts as evidence;
  do not treat a readiness failure as a performance baseline.
- The target startup diagnostic is recorded in
  [`timing-diagnostic.json`](../../../thoughts/performance-audits/target-physical-camera-qualification-20261022/20261022-040555-efa22995/timing-diagnostic.json)
  and its human-readable report. It shows 62/62 freshness rejections and no
  published state; it is not a performance baseline or accuracy result.
