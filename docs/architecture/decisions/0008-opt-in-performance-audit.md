# ADR-0008: Opt-in bounded performance audit

- Status: accepted
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Performance work needs repeatable evidence from the camera publication
boundary through state generation and local client receipt. Vision timing alone
cannot distinguish stale input, queue delay, repeated publications, integration
work, socket delivery, or interference from trajectory and video consumers.
Always-on tracing or synchronous export would add cost to the production
critical path and could turn a slow audit sink into a state-delivery bottleneck.

## Decision

Keep audit instrumentation disabled by default. A validated configuration
setting or process-local `--performance-audit` launch flag activates one or
more independent state, trajectory, and viewer domains. Instantiate recorders,
sampling workers, bounded history, and exporters only after activation.

Use explicit per-frame timing contexts and nested monotonic spans. Preserve
frame wall elapsed time separately from inclusive and exclusive stage costs.
Correlate measurements with process/run identity, host identity, source frame,
timestamp semantics, vision instance, and map/calibration revisions. Treat
shared-memory source timestamps as software publication time rather than sensor
exposure time.

Send audit records through a bounded nonblocking queue to an asynchronous
exporter. Retain bounded samples and event counts, and report sampling, queue,
export-limit, and writer losses. Verify host identity before combining
cross-process monotonic timestamps; otherwise leave the combined interval
unavailable and report measured segments.

Extend the existing recorded-frame benchmark with paced replay for
freshness-oriented behavior and unpaced replay for processing capacity. Use a
headless subscriber for live state delivery and bounded soak tests. Evaluate
accuracy before freshness and throughput. Keep production scheduling,
prediction, and slow-client contracts unchanged until measured evidence and a
separate ADR support a change.

## Consequences

### Positive

- Normal startup avoids audit recorder queues, exporters, resource samplers, and
  detailed frame history.
- State production can be measured without enabling trajectory or viewer
  profiling.
- A slow consumer or exporter can lose audit events but cannot apply queue
  backpressure to state generation; the report makes those losses visible.
- Same-host end-to-end latency uses a verified monotonic clock domain, while
  unsupported or cross-host measurements remain explicit.
- Candidate improvements are ranked with matched workloads and accuracy
  evidence instead of inferred from intuition.

### Negative

- Full audit runs consume CPU, memory, and disk; the sampling rate, retained
  samples, queue depth, resource interval, and event cap must be configured for
  the run duration.
- Absolute shared-memory publication timestamps include software capture and
  publication delay and do not reveal the camera sensor exposure time.
- Existing still labels do not establish temporal identity continuity, cue
  quality, or labeled freshness; those recommendations require additional
  temporal annotations.
- Instrumentation itself must pass repeated off/on comparisons within the
  configured throughput and p95 overhead budget before its measurements are
  trusted.

## Alternatives considered

### Always-on detailed instrumentation

This would make ordinary operation pay for collectors and exporters and could
change the behavior being measured. Detailed audit components are created only
after explicit activation.

### Export synchronously from processing and delivery threads

Disk or consumer delay would then affect the state path. The audit queue is
bounded and producers enqueue without waiting; overload is reported as trace
loss.

### Reuse one shared stage timer

Nested or overlapping operations would overwrite each other and inclusive stage
totals could be double-counted. Explicit frame contexts record both inclusive
and exclusive durations while preserving actual frame elapsed time.

### Select optimizations from profiler timings alone

Fast isolated stages can still cause stale or incorrect state, and changing
scheduling or transport can alter freshness and delivery semantics. Reports
therefore pair critical-path/resource evidence with accuracy and matched
before/after validation.

## Verification

- Deterministic timing tests cover nested, overlapping, incomplete, exceptional,
  and thread-isolated spans; bounded history; and independent audit domains.
- Recorder tests cover default-off activation, process-session identity,
  backpressure drops, export failures, and export limits.
- Replay and live WebSocket integration tests correlate source observation,
  state publication, socket send, and headless receipt while retaining slow
  subscriber isolation.
- The configured formatting, lint, typing, backend test, and coverage gates are
  required before implementation is complete.
