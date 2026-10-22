# ADR-0051: Repeat recorded input for bounded paced audits

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust replay supervisor starts its producer during API startup, before a
performance-audit subscriber connects. The available recorded corpus may be
shorter than the audit warm-up and measured windows, so one-pass replay can
finish before measurements begin. Starting the subscriber earlier does not
solve sustained runs when the source itself is shorter than the requested
window. Repeated frames are useful as a stable load source, but they cannot
stand in for temporal accuracy evidence.

## Decision

Add a runtime-selectable repeat count for paced recorded replay. The default
remains one pass. Repeated cycles receive strictly increasing synthetic source
frame indexes and retain the encoded-image hash, making reused visual content
distinguishable from unique source images in reports. Require a configured
maximum scheduled-frame count and reject runs that exceed either that cap or
the frame-index wire limit before inference begins. The Rust supervisor passes
both settings explicitly. Unpaced replay remains one pass, and repeated input
is labelled as load evidence rather than accuracy evidence.

## Consequences

### Positive

- A short corpus can sustain the subscriber's warm-up and measured intervals
  without changing the source cadence or latest-frame policy.
- Finite replay and the scheduled-frame cap keep source work and result
  retention bounded.
- Existing replay behavior stays unchanged at the default repeat count.

### Negative

- Repeated imagery adds no new temporal, motion, occlusion, or identity evidence.
- Result and detection-batch retention still scale with the configured frame
  cap and existing output-byte limit.

## Alternatives considered

### Add a collector-to-API start handshake

Rejected as the sole fix because it removes the startup race but cannot extend a
finite corpus past the warm-up and measurement window.

### Run replay indefinitely

Rejected because a never-ending child has no finite report boundary and would
require a separate bounded result-retention and log-rotation contract.

### Expand the input directory with copied image files

Rejected because it duplicates storage and hashing work, obscures the original
corpus identity, and does not express the repeat semantics in the runtime
report.

## Verification

- Rust tests verify monotonic frame identity across repeated cycles, source-hash
  reuse, finite schedule counts, and rejection above the configured frame cap.
- Python supervisor tests verify validated configuration and shell-free
  forwarding of the repeat and cap settings.
- A short API replay smoke receives states beyond one corpus duration; the
  existing collector records the finite workload and labels repeated-content
  evidence separately from accuracy claims.
- On the Linux test host, a 30-frame corpus repeated ten times at five frames
  per second completed cleanly and published 221 states out of 300 scheduled
  slots. The short client window reports message and unique-observation rates,
  but server stage timings and cross-host receipt latency are unavailable. See
  the [target replay smoke notes](../../../thoughts/performance-audits/target-rust-replay-smoke-20261021/target-smoke-notes.md).
