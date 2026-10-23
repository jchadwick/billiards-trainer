# ADR-0053: Use five-percent default detail sampling for performance audit

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0085 for collector sampling and report implementation

## Implementation status (2026-10-23)

The 5% ratio applied to detail sampling in the now-retired Python collector.
The maintained Rust audit uses explicit bounded sample retention and does not
apply that fractional sampling default. The old Mac A/B reports remain useful
historical evidence for that collector configuration, but they do not qualify
the current Rust instrumentation overhead. Default-off operation, bounded
retention, and explicit loss disclosure remain required Rust audit contracts.

## Context

The performance audit must retain enough stage and delivery detail to rank
bottlenecks while keeping its opt-in instrumentation overhead within the
configured 2% budget. A three-repetition comparison using 10% detail sampling
on the local M4 Max exceeded that budget. A follow-up used the same 60-second
4K table-video segment for both arms, 5% detail sampling, and three 120-second
measurements per arm. The collector verified matching revision,
dirty-worktree identity, hardware, model, input hash, configuration, and
subscriber settings.

## Decision

Keep audit collection disabled by default. When explicitly enabled, default
both detailed frame sampling and detailed high-frequency event sampling to
5%. Preserve full acceptance for forced correctness and lifecycle events,
bounded queues, and explicit loss accounting. These are audit-only settings;
they do not change normal detection cadence or state publication.

## Consequences

### Positive

- The matched 5% audit run stayed within budget: unique-observation throughput
  changed by +0.22% and source-publication-to-client-receipt p95 changed by
  +0.37%; exporter and client trace losses were zero.
- The run retained 449 detailed frame records and about 570 selected events
  per high-frequency category, enough to rank stable stage costs without
  retaining every event.

### Negative

- Percentiles for rare paths have fewer samples than at 10%, so reports must
  show counts and should extend the run when the affected workload is rare.
- The overhead result is specific to the tested macOS CoreML path and saved
  video. Other runtimes and real cameras still require matched qualification.

## Alternatives considered

### Keep ten-percent detail sampling

Rejected as the default after the first full comparison exceeded the 2%
budget. The matched follow-up at 5% passed; 10% remains configurable for
targeted investigations that need more detail and can tolerate its measured
cost.

### Sample at one percent

Not selected because the matched 5% run met the budget while retaining a more
useful sample for stage percentiles and delivery categories.

## Verification

- The former Python tests verified runtime-selected sample configuration,
  source freshness fallback, and observation accounting. They were retired
  with the collector and are historical, not a current acceptance gate.
- The matched off/on reports are
  [`20261021-212120-6a414523`](../../../thoughts/performance-audits/local-sampled-instrumentation-ab/matched-5pct-config-refresh/20261021-212120-6a414523/report.md)
  and
  [`20261021-212922-98f45b17`](../../../thoughts/performance-audits/local-sampled-instrumentation-ab/matched-5pct-config-refresh/20261021-212922-98f45b17/report.md).
  The on report records `status: pass`, three repetitions, matching inputs and
  worktree, and zero export loss.
