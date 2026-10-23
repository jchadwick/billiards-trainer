# ADR-0048: Isolate trajectory workload during state-path audits

- Status: superseded
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0095

> Superseded by ADR-0095: this Python-service audit control was never ported
> to the supported Rust server. The decision and verification below are
> historical and do not describe a current setting or Rust behavior.

## Context

The state-only audit domain controls which trajectory measurements are
collected; it does not control whether the integration service still admits
and computes trajectory predictions. A state-only run therefore includes
prediction scheduling and can misattribute its CPU and latency cost to primary
ball-state production. Production prediction behavior and its lifecycle must
remain unchanged unless a separate production decision is made.

## Decision

Add performance_audit.trajectory_workload_enabled, defaulting to true. Honor
false only while an audit recorder is active. The integration service
continues applying observations to Core and refreshing phase/calibration
prediction lifecycle state, but it does not admit new prediction work, clears
queued work, and invalidates any in-flight result when that audit control is
active. The existing trajectory_enabled setting remains an independent
measurement switch. Normal startup, including normal startup with audit
collection disabled, retains existing prediction behavior.

For a matched workload pair, use the same input, model, detector settings,
calibration, stream subscriptions, and audit domains; change only
trajectory_workload_enabled. Use false for the state-production arm and true
for the trajectory-work arm. Collect trajectory detail in a separate
instrumented run if its own timing overhead would otherwise differ between
the pair.

## Consequences

### Positive

- State-only measurements can exclude prediction admission and compute while
  preserving accepted state updates and prediction invalidation semantics.
- The production default and trajectory lifecycle are unchanged outside an
  explicitly enabled audit.

### Negative

- Workload comparisons require restarting the service with the alternate
  effective configuration, and the two arms must be matched manually by the
  collector's recorded provenance.
- The disabled arm still pays for the small lifecycle-refresh work needed to
  preserve phase and calibration correctness.

## Alternatives considered

### Change production prediction scheduling to improve state throughput

Rejected because the existing measurements do not establish that production
prediction work should be removed, deferred, or rescheduled. This decision
creates an audit control only; any production scheduling change requires
accuracy evidence and its own ADR.

## Verification

- backend/tests/unit/test_integration_observation_freshness.py verifies that
  audit workload-off still updates Core and refreshes lifecycle state while
  suppressing prediction admission.
- backend/tests/unit/test_config_validation.py verifies the enabled default
  and validates the new setting through the shared configuration schema.
- backend/tests/unit/test_prediction_lifecycle.py remains the regression
  coverage for production lifecycle behavior.
