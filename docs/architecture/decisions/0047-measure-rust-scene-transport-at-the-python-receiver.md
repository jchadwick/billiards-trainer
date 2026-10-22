# ADR-0047: Measure Rust scene transport at the Python receiver

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust scene producer writes compact JSON snapshots to a bounded loopback
stream. The Python receiver already validates each complete payload and uses a
one-item latest-only queue. The performance audit needs payload-size and queue
pressure evidence to separate inference capacity from local delivery effects.
The producer's versioned timing side channel is strict, and extending it would
require another protocol version and duplicate facts the receiver can measure
directly. Normal runtime must not create audit buffers or retain per-frame
transport history.

## Decision

Measure scene transport at the Python receiver. Keep transport auditing
disabled by default. The opt-in qualification enables scalar counters only
after warm-up and snapshots them at the end of the measured window. Report
accepted snapshot count, total and min/mean/max payload bytes, sequence gaps,
queue depth before enqueue (zero or one), replacement count, replacement
fraction, queue capacity, and observed high-water depth. Do not retain
per-snapshot payload or queue events and do not change the producer wire
protocol, latest-only queue, or delivery semantics. Report the measured window
and local monotonic clock explicitly.

## Consequences

### Positive

- The measurement uses the exact serialized bytes and queue state seen by the
  consumer, so it includes producer writes and local transport effects without
  comparing unrelated process clocks.
- The opt-in window excludes warm-up, and disabled mode creates no audit
  collector, event buffer, callback, or exporter work.
- The Rust producer and its strict timing-record schema remain unchanged.

### Negative

- Receiver measurements cannot split producer serialization, socket write,
  kernel buffering, and consumer parsing without further producer-side spans.
- Aggregate size statistics do not preserve per-frame distribution or correlate
  individual payload sizes with specific inference stages.
- Queue occupancy is constrained to zero or one by the existing contract; the
  counters diagnose replacement pressure but cannot evaluate a deeper queue
  policy.

## Alternatives considered

### Add payload and queue fields to every Rust timing record

Rejected for this audit increment because it requires another strict side
channel schema version and reports facts already observable at the receiver.
Consider it only if evidence shows producer-side separation is necessary.

### Retain a per-snapshot transport trace

Rejected because aggregate bounded counters answer the current capacity and
pressure questions without retaining another frame-indexed trace or increasing
audit exporter traffic.

## Verification

- Ingress unit tests enable auditing after startup, validate payload byte
  aggregation and queue replacement counters, and confirm the disabled path
  remains inactive.
- Rust live supervisor tests verify the audit can be enabled and finalized
  through the qualification interface without changing normal status.
- Collector tests verify the measured transport summary is included in JSON and
  Markdown qualification output.
- Validate all architecture decision records with the repository ADR validator
  and run the focused ingress, supervisor, and collector tests.
