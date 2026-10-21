# ADR-0041: Bound opt-in live pipeline timing export

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The opt-in Rust live path can deliver fresh scene snapshots, and the Python
audit collector measures local state receipt, ordering, gaps, and bounded export
loss. A matched development-Mac instrumentation comparison completed with three
120-second runs per condition: throughput was 19.413 states/s off and 19.253
states/s on (0.82% point-estimate reduction); mean p95 receipt interval was
54.371 ms off and 54.886 ms on (0.515 ms increase). No audit records were lost.
The run-to-run variation and small number of repetitions make the 2% overhead
budget inconclusive rather than proven.

Those state-delivery measurements do not rank the target detector stages. A
separate Linux target CPU replay of the recorded-image inference executable
measured 320.7 ms mean ONNX inference, 56.4 ms mean preprocessing, and 59.8 ms
mean JPEG decoding. The replay was not a live camera session and did not run the
API or client. It reinforces the need for stage-level evidence but does not
qualify the live timing side channel or establish target end-to-end latency.
`CapturePipelineResult` contains model preprocessing, inference, decode/NMS,
calibration, host-read to inference-start, and host-read to scene-queue
durations. Tracking/state assembly and scene serialization also need separate
explicit spans before their costs can be ranked. Scene transport has cumulative
queue/write statistics but no per-frame correlation.

The main scene channel carries application data and must retain its current
schema and freshness behavior. Audit collection is optional, must never block
camera capture or localization, must retain no RGB frames, and must remain
bounded during a long run. The development API and internal local streams have
no authentication, per the project decision; this audit channel must not add
credentials or an authentication handshake.

## Decision

Add an opt-in, versioned Rust live-audit side channel separate from the public
scene stream. When disabled, the live executable creates no audit queue, writer
thread, listener, samples, or exporter. When enabled by the local qualification
runner, the Python supervisor starts a loopback-only receiver before launching
the child and passes its endpoint and validated bounds explicitly.

The live process emits length-prefixed JSON records through a bounded queue to
a dedicated writer thread. The per-frame hot path uses non-blocking enqueue;
queue overflow increments a loss counter and cannot delay or fail scene
publication. The writer uses configured connect/write deadlines and bounded
message size. Records contain run/process/session/frame/transport identity,
timestamp kind and stage, configuration/calibration revisions, measured stage
durations, and detection/track counts. They contain no image or RGB payload and
do not duplicate ball coordinates. Durations are explicit spans: preprocessing,
model inference, decode/NMS, calibration, tracking/state assembly, scene
serialization, and scene-queue submission. Missing measurements are reported
as unavailable; `0` is reserved for an observed zero duration.

On graceful shutdown, the child sends a final versioned summary with pipeline
skip/error counters, scene transport counters, exporter loss, and whether the
stream ended cleanly. If the child is killed or the footer is absent, the
collector marks the run partial and does not infer zero loss. Python correlates
audit records with received states by engine, source-session, source-frame, and
transport-sequence identity, then reports unmatched and dropped records beside
scene ingress and publication counts. This path is available only to the
explicit performance-audit tool; normal provider selection and the public scene
schema remain unchanged.

The receiver binds only to loopback and uses an ephemeral port. No
authentication, TLS, tokens, or authorization contract is added. Identifiers
bind records for measurement and provenance, not as security credentials; local
processes can therefore spoof audit data. Audit results are diagnostic evidence
and are not a security boundary.

## Consequences

### Positive

- Stage timings and loss counters become available without putting diagnostics
  in client-facing scene payloads.
- A slow or disconnected audit receiver can consume only the bounded queue and
  writer resources; it cannot stall the detector or latest-state publisher.
- The same record identity lets the report correlate internal compute cost with
  state receipt, freshness, and transport gaps.
- Disabled operation preserves the current runtime and has no telemetry thread,
  queue, or socket allocation.

### Negative

- The side channel adds protocol, receiver lifecycle, and shutdown/footer
  handling across Rust and Python.
- Measurement and record construction have non-zero cost when enabled; matched
  audit-off/audit-on runs must quantify that overhead before results are used
  to rank small optimizations.
- A bounded queue can lose timing records under load. Reports must expose every
  loss and mark incomplete or hard-killed runs partial.
- Loopback transport is unauthenticated by design, so records are not protected
  from another local process; only the opt-in developer audit uses this channel.

## Alternatives considered

### Put timings in public scene snapshots

Rejected because diagnostic fields would change a retained application
contract, increase every consumer's payload and parsing work, and couple
performance measurement to state publication. A separate side channel keeps
normal client behavior unchanged.

### Write per-frame records directly to stderr

Rejected as the structured data path because the supervisor currently retains a
bounded diagnostic tail rather than a complete correlated event stream. High
rate records could evict actionable errors and still could not safely apply
backpressure to the detector. Stderr remains for bounded diagnostics.

### Omit live stage timings

Rejected because state-receipt cadence alone cannot rank inference, geometry,
tracking, serialization, or transport as bottlenecks. Without per-frame spans,
optimizations would be selected from intuition rather than evidence.

## Verification

- Rust unit tests prove that disabled mode allocates no sink, the record schema
  carries source identity and timestamp semantics, explicit spans separate
  preprocessing/inference/decode/calibration/tracking/serialization, skipped
  outcomes remain distinct, and a full queue increments loss without blocking
  the publishing thread.
- Rust process tests exercise a slow or disconnected loopback receiver,
  message-size/deadline bounds, graceful footer and counter delivery, and
  forced process exit with an absent footer.
- Python supervisor/collector tests reject malformed and oversized records,
  correlate matching scene and timing identities, report unmatched records and
  exporter loss, preserve partial status without a footer, and verify that no
  RGB/image bytes are stored.
- With the option disabled, run the same live tests and confirm no audit
  socket, queue, task, or worker is created. Run matched live audit-off/on
  repetitions; instrumentation is trusted only within the configured overhead
  budget and with a complete, loss-free or explicitly loss-qualified sample.
- The existing API provider test and public scene fixtures continue to pass,
  demonstrating that qualification telemetry has not enabled provider routing
  or changed the application stream.
- Run evidence (2026-10-21): the development-Mac A/B point estimates were within
  the 2% budget, with zero observed audit loss, but three runs per condition are
  insufficient to qualify the bound under measured run-to-run variation. The
  target Linux CPU image replay is summarized in
  [the target report](../../../thoughts/performance-audits/rust-live-trained-model-export-20261021/target-linux-replay-20261021/report.md);
  it is not live-camera or API/client overhead evidence. Target live overhead
  remains unqualified.
