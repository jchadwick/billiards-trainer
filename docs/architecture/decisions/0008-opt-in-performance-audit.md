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

## Historical implementation status (2026-10-21)

This records the earlier Python-backed runtime and is retained as historical
context. Those collector implementations and reducer commands were later
retired during the Rust migration; this section does not describe the current
codebase.

The bounded API, vision, and video-process collectors, deterministic nested
frame spans, headless receipt audit, and repeatable report exporter are
implemented. High-frequency state detail is selected deterministically before
event allocation and timing; 10% sampling retains exact unique-observation
counts and forces lifecycle/correctness anomalies through the bounded queue.
The video queue-retrieval interval is explicitly unavailable because frame
identity is only known after the nonblocking read. Viewer detail is selected
deterministically by source writer/frame before JPEG and send timing; only the
small read clocks needed to discover that identity run on unsampled frames.
The headless collector bounds source-identity history and reports evictions.
Trajectory timing summaries include successful, failed, stale, and
publication-failed outcomes while retaining separate outcome counts. Resource
probe failures are nonfatal and clear stale values. The end-to-end replay test
now traverses detection, integration conversion, Core update, state publication,
WebSocket send, and client receipt.

The local M4 Max matched 3×30-second screen passed the configured overhead
comparison: 24.55 unique observations/s with instrumentation off versus
25.07/s with sampled state audit, and 61.83 ms versus 60.39 ms p95
publication-to-client receipt. Both exporters reported no dropped or failed
records. The Linux saved-image cadence screen processed 120/120 frames at 3 FPS
with no pending replacements; this does not establish temporal accuracy or
camera-mode support. These short runs qualify the sampler directionally, not
the default 3×120-second overhead gate, the 20-minute memory bound, detector
accuracy, or target-camera behavior.

The latest repository run passed 1,750 tests with two model-dependent skips.
Ruff, Black, strict mypy, Rust formatting, Rust lint, the Rust tests, and all 51
ADR validations pass. Pytest still exits nonzero because total coverage is
56.48%, below the configured 80% gate; the coverage shortfall remains an open
repository-wide acceptance item.

## Current Rust implementation status (2026-10-23)

The Rust live capture pipeline now creates a bounded per-frame timing context
only when live pipeline timing audit is enabled. It records host read-completion
to scene-queue-submission elapsed time and nested inclusive/exclusive spans for
detector preprocessing, model execution, decode/NMS, calibration, tracking and
state assembly, serialization, and queue submission. The audited receive path
also times capacity-one mailbox residence from host read-completion to dequeue
and reports it independently from the remaining freshness, admission, and
scheduling work before inference. Cross-process intervals remain segmented
unless monotonic clocks are verified comparable. Versioned frame records
retain the span tree, and the summary reports frame elapsed and per-span
percentile distributions with missing, invalid-clock, and dropped span counts.
Deterministic tests cover nesting, overlapping children, error and panic
cleanup, cross-thread completion, invalid clocks, bounded capacity, and
disabled contexts.

The first Rust audit off/on comparison stayed inside the 2% budget for its
development-Mac, looped-video workload, but its executable predates the mailbox
residence metric. Repeat that comparison with the current build and output
outside the repository. Other open gates are independent runtime controls and
matched interference runs for trajectory computation; a repeatable video-viewer
workload comparison; gateway/WebSocket stage coverage; and reviewed temporal
physical-accuracy labels for moving, occluded, settling, and interrupted-camera
cases. The `--include-trajectory` option adds trajectory receipt collection
only; it does not toggle trajectory computation. Historical Python audit
results remain context, not current-Rust qualification.
