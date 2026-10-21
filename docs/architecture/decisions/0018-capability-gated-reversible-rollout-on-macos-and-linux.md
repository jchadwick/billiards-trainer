# ADR-0018: Capability-gated reversible rollout on macOS and Linux

- Status: proposed
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust candidate introduces a new inference/capture runtime while existing UI,
API and secondary engines remain. macOS and Linux differ in camera backends,
accelerators and native packaging. A working CPU fallback does not prove the
same latency on all machines. The legacy provider also has known limitations
and cannot inherit new accuracy guarantees from transport compatibility.

## Decision

Migrate through explicit selectable Python/Rust providers, recorded-input
qualification, then exclusive live acquisition and target-platform trials.
Keep Python selected until the candidate passes documented capability and
release gates. No silent automatic provider failover or concurrent camera owners.

Maintain a qualification matrix by provider, camera mode, platform,
model/export/backend and optional workload. Record which guarantees actually
pass: image observation, physical precision, moving coverage, identity, cue,
prediction and preview. An unsupported guarantee is degraded/unavailable and
operations depending on it are blocked; rollback does not advertise untested
new capabilities.

Require accuracy/freshness/compatibility, lifecycle, sustained memory and
interference tests, plus a repeatable freshness or unique-observation-capacity
improvement against the corrected Python reference before performance-driven
default cutover. Use configurable warm-up/repeated measured/soak windows from
the audit plan and validate actual execution placement.

Exercise rollback by stopping/releasing the active camera owner before starting
the other. Retain a qualified legacy provider during a stabilization window
chosen from deployment evidence, then remove obsolete production paths while
preserving useful reference tools and fixtures.

## Consequences

### Positive

- Makes rollout reversible without discarding current working application.
- Exposes platform/capability limitations honestly.
- Separates replay readiness from live camera/preview/calibration readiness.
- Avoids deploying a rewrite merely because it compiles or reports higher FPS.

### Negative

- Two providers temporarily increase test and dependency maintenance.
- Linux CPU support may be correct while insufficient for the desired timing.
- Qualification needs access to actual target hardware/cameras.
- Stabilization and rollback operations require explicit operational ownership.

## Alternatives considered

### Replace the default immediately after the first Rust inference

That verifies neither physical accuracy nor frontend/camera lifecycle and
makes recovery harder.

### Promise identical performance from cross-platform source compatibility

Native backends, accelerators and camera modes differ. Qualify profiles and
publish measured results per target.

### Retain the legacy implementation indefinitely without capability limits

This accumulates maintenance and can silently reintroduce stale/incorrect
semantics. Bound its production lifetime by evidence and preserve reference
evaluation separately.

## Verification

- macOS/Linux build, unit, contract, real-model replay and target-camera tests pass.
- Both providers retain public compatibility and truthful capability status.
- Three measured runs after warm-up and a separate bounded-memory soak meet
  declared configurable accuracy/freshness/resource gates.
- Optional-consumer and slow-client runs preserve primary guarantees.
- Worker death, camera interruption, calibration replacement, shutdown and actual
  provider rollback are exercised without overlapping acquisition owners.
- Deployment packages record native dependencies, model manifest and actual backend.

## Revisit conditions and related records

Keep Rust optional when evidence does not establish the intended improvement.
A different rollout topology or removal of rollback requires an explicit
record of capability evidence and operational consequences. Preserve ADR-0005
single-table ownership and ADR-0003 slow-client delivery.


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 4, 5 and 8.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/main.py](../../../backend/main.py).
- [Existing backend/tests/integration/test_pipeline_replay.py](../../../backend/tests/integration/test_pipeline_replay.py).

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.
