# ADR-0095: Defer trajectory interference controls until a Rust workload exists

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0048
- Superseded by: None

## Context

ADR-0048 proposed `performance_audit.trajectory_workload_enabled` for the
Python integration service. Its decision and verification were not migrated
when Rust became the supported server runtime. The current Rust live service
produces detected ball state and does not run or publish a trajectory
prediction workload. The `--include-trajectory` option in the Rust audit
controls receiver allocation and subscription only; it cannot enable or
disable server computation. The old Python implementation is no longer a
valid basis for a current Rust A/B test.

The primary live objective is correct, fresh ball locations. Trajectory
prediction is secondary. Adding a switch to an audit collector without an
active producer would create an untestable configuration and misleading
experiment rather than evidence about the primary path.

## Decision

Do not implement a trajectory workload control in the Rust server or describe
trajectory receipt collection as a workload-isolation experiment. Keep the
live audit focused on detector-to-state correctness, freshness, unique source
observations, delivery, and measured interference from workloads that actually
run.

When a Rust trajectory/prediction engine becomes an active supported workload,
first define its lifecycle, state dependency, and consumer contract. Then add a
matched state-path versus trajectory-work experiment with independently
measured prediction output, bounded scheduling, and regression coverage for
state freshness and correctness. Record that implementation in a new ADR
before using it to rank production changes.

## Consequences

### Positive

- Avoids shipping a no-op setting or claiming a trajectory-isolation result
  that the supported Rust server cannot produce.
- Keeps current audit evidence aligned with the Rust runtime and the primary
  detection/localization objective.

### Negative

- The historical Python trajectory-interference hypothesis remains untested
  against the current Rust server.
- A future prediction feature will need its own scheduling, lifecycle, and
  interference design before it can be independently audited.

## Alternatives considered

### Port the Python audit switch preemptively

Rejected because there is no Rust production trajectory workload to control.
The switch would not isolate measurable work and would encourage invalid
comparisons.

### Treat `--include-trajectory` as the isolation control

Rejected because it only changes what the subscriber listens for and records;
it does not change server behavior or computation.

## Verification

- Confirm the Rust server has no trajectory prediction producer or
  `trajectory_workload_enabled` setting.
- Confirm the audit documentation distinguishes receiver subscription from
  server workload control and does not claim current trajectory isolation.
- Revisit this decision when a supported Rust trajectory workload and output
  contract are introduced.
