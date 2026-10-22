# ADR-0049: Make performance audit count semantics explicit

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0050

## Context

High-frequency pipeline and publication events are selected deterministically,
then passed through a bounded asynchronous exporter. Detail sampling,
probabilistic export sampling, queue overflow, and export limits can all reduce
the records retained by a report. Counting retained events as total processed
or published observations therefore understates workload and can imply exact
duplicate rates that the trace cannot support. Viewer experiments also need to
select the same replay frames across process restarts to make A/B samples
comparable.

## Decision

Use the source frame number as the deterministic shared-memory viewer sampling
identity. Keep writer process identity in event provenance, not the sampling
key, so an identical replay frame sequence selects the same detail sample
after a service restart. The raw-camera route has no shared source-frame
identity and continues to use its per-client sequence.

Label server-side event-derived totals as recorded or sampled detail counts.
Do not present counts calculated from retained high-frequency events as exact
pipeline throughput or repeat rates. Keep headless subscriber state
accumulation as the source for delivered-state throughput and repeated
publication reporting, including its configured bounded identity window.
In-memory profiler counters continue to report exact total and failure counts.
Version the collector result schema as version 4 to make these semantics
explicit to consumers.

## Consequences

### Positive

- The report distinguishes retained detail records from complete state
  delivery measurements and cannot treat sampling loss as missing work.
- Viewer detail samples remain paired across identical replay frame sequences
  even when the writer process is restarted.
- No per-frame exact-count event is added to the high-frequency producer path.

### Negative

- Server-side high-frequency counts cannot answer exact unique processed
  observation throughput; use the headless subscriber for delivered states
  and profiler counters for total frames.
- Cross-run viewer sampling is paired by frame number only when the replay
  starts from the same source sequence; camera captures are not content-matched
  across separate sessions.

## Alternatives considered

### Emit an exact audit event for every processed and published observation

Rejected because an event per state would bypass the existing detail sampling
and add queueing, lock, and export work to the timing path whose overhead the
audit is meant to bound.

### Include writer PID in the viewer sampling identity

Rejected because process IDs change on restart, causing otherwise identical
replay frames to receive different sample decisions.

## Verification

- Collector tests verify server event-derived counts use recorded-detail names
  and disclose their sampling semantics.
- Viewer-stream tests verify the source frame number is used as the sampling
  identity independently of writer PID.
- Run the focused collector and stream tests, the backend suite, and the ADR
  validator.
