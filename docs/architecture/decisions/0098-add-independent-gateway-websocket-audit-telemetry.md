# ADR-0098: Add independent gateway WebSocket audit telemetry

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The live detector audit measures capture, inference, state-envelope
serialization, and the detector's loopback scene write. The headless subscriber
measures its own WebSocket receipts. The Rust gateway currently measures none
of its scene ingress, latest-state publication, per-client selection, public
message assembly, JSON serialization, or WebSocket send call. These missing
stages prevent a run from distinguishing detector delay from gateway/client
delivery cost.

The existing detector audit stream is process-specific: its footer describes
the detector, and its producer PID drives detector resource sampling. Adding
gateway records to that stream would mix process identities, overwrite its
single-footer summary, and could attribute gateway resources to the detector.
Gateway timing must therefore have an independently enabled, bounded audit
stream. Normal server startup must create no gateway collector, client identity
counter, queue, worker, or exporter.

## Decision

Add an independent gateway WebSocket audit switch and a separate bounded
loopback receiver/exporter. Emit scalar timing records keyed by audit run ID,
gateway process and instance ID, ephemeral client ID, public publication
sequence, producer engine/session/frame identity, source transport sequence,
and available configuration/calibration revisions.

Use the gateway's own monotonic clock for state publication-to-client
selection, watch wake-to-consume, message assembly, JSON serialization, and
the deadline-bounded WebSocket `send` call. Name the latter as an application
send-call duration; it does not isolate kernel buffering or prove physical
client receipt. Include payload size, send outcome, repeated publications,
coalesced publication gaps, and source-sequence gaps. Preserve latest-only
selection, the existing deadline and disconnect behavior, and the public
WebSocket message schema.

Create gateway audit state only when explicitly enabled. Submit through the
existing nonblocking bounded record queue, dropping audit records and counting
loss on saturation. Keep gateway events and footer separate from detector
timings and detector resource sampling. Report gateway and subscriber clock
segments separately; one-way publication-to-receipt latency remains
unavailable unless the clock domains are explicitly verified comparable.

## Consequences

### Positive

- Gateway serialization and per-client send cost can be distinguished from
  detector compute and subscriber receipt cadence.
- Separate bounded streams preserve detector footer, PID/resource attribution,
  and existing latest-only delivery behavior.
- Audit loss and unavailable clock comparisons remain explicit in reports.

### Negative

- Enabled audit records allocate identity strings and scalar event data and
  consume bounded CPU, memory, and loopback bandwidth. Matched audit-off/on
  runs must qualify that overhead before small-stage comparisons are trusted.
- The application send future cannot distinguish WebSocket buffering, kernel
  write completion, and remote client processing. Subscriber-local receipt
  remains a separate clock segment.

## Alternatives considered

### Merge gateway events into the detector audit stream

Rejected because detector and gateway process IDs, resource sampling, and
shutdown footers have different ownership. Mixing them would make the existing
single-producer footer and detector PID semantics ambiguous and would couple
gateway measurement to detector instrumentation.

### Put gateway timing in public WebSocket messages

Rejected because this changes the application contract, increases every
client's payload and parsing work, and measures clients differently depending
on their behavior. Diagnostics remain on a private opt-in audit stream.

### Infer gateway cost from subscriber receipt intervals

Rejected because receipt intervals cannot isolate gateway selection,
serialization, and send-call work, and receiver monotonic ticks are not
verified comparable to the server's clock.

## Verification

- Gateway unit tests verify disabled mode creates no exporter, accepted records
  preserve source/publication/client identity and clock-domain labels, and
  saturated audit queues drop records without waiting on the exporter.
- WebSocket integration tests verify latest-only gap and repeated-publication
  accounting, success/deadline/error outcomes, and that a slow client remains
  isolated while a responsive client receives the newest state.
- Runner tests verify the independent gateway flag works without detector
  timing, gateway and detector receivers retain separate process footers and
  loss counts, and per-window JSON/JSONL artifacts report sample counts,
  percentiles, losses, and unavailable cross-process one-way latency.
- `make test-rust`, `make lint-rust`, and ADR validation pass. A matched
  audit-off/on live run must remain within the 2% throughput and p95 latency
  overhead budget before gateway timing is used to rank production changes.
