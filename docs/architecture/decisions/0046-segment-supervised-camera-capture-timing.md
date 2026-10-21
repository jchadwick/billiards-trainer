# ADR-0046: Segment supervised camera capture timing across process boundaries

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The GStreamer helper runs in a child process and transfers owned RGB frames to
the Rust live process over a framed pipe. The helper currently emits an elapsed
monotonic value that the parent validates and discards. The parent then records
`Instant::now()` after it has read and copied the complete frame record. Its
existing read-to-inference duration therefore excludes appsink waiting, RGB
conversion, helper serialization/write, pipe transfer, and parent parsing and
copying. Treating that value as camera-to-inference latency is incorrect.

Helper and parent `Instant` values have independent process-local origins. The
GStreamer pipeline PTS is not a verified camera-exposure timestamp. Capture
timing must be opt-in, bounded to scalar durations, and must not add work when
the live audit is disabled. The public scene payload remains unchanged.

## Decision

When live pipeline audit is enabled, the helper will report bounded duration
segments for appsink pull wait, RGB buffer allocation, and RGB mapping/copy. The
parent will measure the complete framed-record read using its own monotonic
clock. That parent span includes pipe transfer and any producer backpressure,
but does not attribute a separate helper write/flush duration. Each duration is
measured and reported within the process that observed it; no helper monotonic
timestamp will be subtracted from a parent timestamp.

The parent frame envelope will retain the parent monotonic completion instant
after the complete frame record has arrived. The existing interval to inference
start will be named and documented as parent-record-read-completion to
inference-start, including receiver queue wait. Camera exposure and source
pipeline arrival remain unavailable unless a future platform source supplies a
verified common clock or explicit GStreamer instrumentation.

The helper frame-header protocol will add a versioned timing form. The parent
will continue accepting the current untimed frame-header form. The separate
Rust live audit record will advance to schema version 2 and include optional
capture durations; `null` means that a source did not measure that stage. No
RGB payload or coordinate data is added to the timing stream.

## Consequences

### Positive

- Reports can distinguish helper pull/conversion/pipe-read work from inference
  scheduling delay instead of presenting a partial interval as end-to-end.
- Process-local clock domains remain explicit, so measurements are not
  contaminated by comparing unrelated monotonic origins.
- The metrics can be correlated with the existing session/frame identity in
  the opt-in live audit while leaving normal scene messages unchanged.
- Disabled audit mode avoids the new duration sampling and timing fields.

### Negative

- The helper frame-header and Rust audit schemas need version-aware parsing and
  tests across old untimed and new timed records.
- Timing spans cover application boundaries, not camera sensor exposure or
  every internal GStreamer element. `appsink_pull_wait` is wait time, not total
  upstream camera-pipeline work.
- Helper serialization and write/flush are not split out from the parent framed
  read span. Reporting a self-measured write duration inside the same frame
  record would require a second timing record or a delayed association scheme.
- The measurements add some work when audit is enabled; the existing audit
  overhead budget and bounded-loss checks still apply.

## Alternatives considered

### Subtract helper elapsed time from the parent's monotonic time

Rejected because each process has its own `Instant` origin. Converting either
value to a wall clock would add clock adjustments and synchronization error.

### Call the parent read completion time camera capture time

Rejected because it occurs after helper encoding, pipe transfer, parent parsing,
and RGB payload copy. It is a sound boundary for downstream age measurement,
but not an exposure or driver-read timestamp.

### Add a second helper timing record for write and flush duration

Deferred because the current qualification needs a bounded application-level
capture split first. A second per-frame record would increase helper protocol
and correlation work; the parent framed-read span already includes transfer
and producer backpressure from the receiver's view.

### Put capture timing in the public scene snapshot

Rejected because these details are audit diagnostics, not application state.
The opt-in side channel already provides bounded, correlated timing records.

## Verification

- Helper tests assert that timed mode reports each supported duration and that
  untimed mode keeps the legacy record form without timing values.
- Parent source tests verify a timed helper record carries the durations and a
  parent framed-read span, while its completion `Instant` is captured only
  after the full RGB payload is read.
- Audit serialization and Python validation tests cover schema version 2,
  nullable unavailable spans, malformed duration rejection, and the legacy
  schema version 1 form.
- Live tests confirm audit-disabled source launch does not pass the timing flag.
- Run the Rust perception tests and Python audit collector/supervisor tests;
  use the existing instrumentation-off/on comparison before relying on capture
  timing for fine-grained rankings.
