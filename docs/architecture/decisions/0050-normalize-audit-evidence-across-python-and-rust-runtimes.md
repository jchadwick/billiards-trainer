# ADR-0050: Normalize audit evidence across Python and Rust runtimes

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: ADR-0049
- Superseded by: ADR-0085 for cross-runtime collector implementation

## Implementation status (2026-10-23)

ADR-0085 retired the Python collector and its Python/Rust envelope-normalizing
path. The evidence semantics remain in force for the Rust audit: distinguish
fresh observed frames from cached or failed updates, preserve native source
identity, disclose unavailable measurements, and never subtract timestamps from
unverified clock domains. The maintained audit now consumes Rust state
directly; it does not support Python envelope normalization or cross-runtime
overhead comparison. The historical collector schema and its tests are not a
current output contract.

## Context

The application can publish state from the legacy Python vision path or the
native Rust scene path. Python state envelopes identify frames with numeric
frame numbers, process IDs, and vision instance IDs. Rust scene envelopes use
engine instance, source session, and source frame IDs, with source ticks and
clock semantics carried in the observation status. The existing headless audit
collector recognized only the Python shape, so valid Rust deliveries appeared
uncorrelated and incomplete. Containerized collection also has no Git checkout
or local paths corresponding to the configured model and replay assets; treating
those absent values as clean or identical can produce false comparability.

## Decision

Normalize both state envelope shapes in the collector while preserving each
source's original identity and timestamp semantics. For Rust scenes, retain the
`(engine_instance_id, session_id, frame_id)` identity, derive a numeric ordering
counter only from the frame ID's numeric suffix, and count a message complete
when it includes the stream sequence, ball list, observation status, and a
usable source identity. Do not reinterpret Rust source ticks as wall time or
subtract them from a different monotonic clock domain.

Count a new source frame as a fresh ball observation only when its ball-position
status is explicitly `observed`. Count cached, failed, skipped, and unavailable
source-frame updates separately from fresh detections. Preserve a separate
total for all unique source-frame updates, including non-fresh updates.

Keep shared-memory viewer detail sampling keyed by source frame number, with
writer process identity recorded only as provenance. Keep high-frequency
server totals labeled as recorded or sampled detail counts. Version the
collector summary schema as version 6 for the bounded source-identity
deduplication window and explicit retained-event semantics alongside the
fresh-observation and source-frame-update metrics.

Build report provenance from the configured active perception provider. Rust
replay runs record hashes for the configured ONNX model, source checkpoint,
inference executable, and replay input (including deterministic directory
hashes). Python runs continue to use their configured model and video input. If
the collector cannot inspect a Git checkout, report revision and dirty-worktree
identity as unavailable; overhead comparisons require known, matching revision,
worktree, model, and input identities.

## Consequences

### Positive

- Rust and Python deliveries use one receiver report without changing either
  runtime's public state envelope.
- Correlation, completeness, source ordering, model identity, and replay input
  are represented using the active runtime's real contract and configuration.
- Cached state and failed updates cannot inflate the primary fresh-observation
  throughput metric.
- Unavailable repository or clock identity cannot be mistaken for a clean,
  comparable result.

### Negative

- The collector must maintain normalization for both state contracts as either
  runtime evolves.
- Rust source clocks may not be comparable with the receiver clock; in those
  cases end-to-end latency remains unavailable even when freshness and receipt
  counts are measurable.
- A finite replay that starts before the subscriber connects cannot establish
  steady-state rates. Such a run is a startup smoke only and must be labeled as
  such until orchestration captures the full replay window.

## Alternatives considered

### Add Python-specific identity aliases to Rust state messages

Rejected because it would couple the Rust wire contract to legacy Python
process fields and duplicate identity data solely for one measurement consumer.
The collector can normalize the two existing contracts at its boundary.

### Treat missing Git metadata as an empty, clean worktree

Rejected because two unknown revisions or worktrees do not establish that the
measured builds match. Unknown identity must make A/B overhead comparisons
inconclusive.

## Verification

- Before the collector was retired, its unit tests covered legacy and Rust
  envelopes, complete/repeated observation accounting, cached-versus-fresh
  throughput, timestamp-stage preservation, replay/model/executable hashes,
  and unavailable Git identity. These are historical differential tests, not
  a maintained Python compatibility requirement.
- The target API smoke artifact records a native binary hash, model and source
  checkpoint hashes, replay-directory hash, Rust source identities, and the
  limitation that the finite replay outlasted the measurement's active source
  window.
- Rust audit tests and the repository checks validate the current native
  identity, freshness, timing, and provenance behavior; do not run the retired
  collector tests as a current acceptance gate.
