# ADR-0016: Prioritize localization and isolate optional consumers

- Status: proposed
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Fresh physical ball locations are primary. The existing integration path
calculates/adopts predictions alongside state updates, and optional MJPEG viewers
perform repeated synchronous copy/encode work on the API event loop. The existing
phase scheduler can suppress detailed localization during many-ball motion.
Its default-off state means that policy did not cause the audited baseline,
but it conflicts with the new product priority.

## Decision

Give localization its own freshness/coverage budget and publish eligible scene
observations independently of cue analysis, number identity, physics, preview
and recording. Optional consumers receive immutable snapshots or owned image
views through bounded latest-work interfaces and explicit workload enable
settings separate from audit switches.

Maintain localization coverage in dense motion. Reduce optional work before
reducing primary observations. Use global reconciliation/recovery to prevent
ROI or static-marker exclusion from creating permanent physical blind spots.
Specific rates/resolutions are configured from labeled measurements.

Preserve existing prediction detached inputs, one-active/one-newest-pending
coalescing, revision/age rejection, shutdown draining and frozen-shot comparison
lifecycle. Correct source-age semantics under ADR-0010. Initially retain Python
physics behind a separately scheduled scene consumer; test dedicated workers
or processes where resource evidence warrants them.

Encode optional preview outside primary processing/API scheduling, once per
bounded profile, and fan out newest encoded bytes. Slow clients do not request
repeated per-client encoding or retain historical preview work.

## Consequences

### Positive

- Primary locations remain useful during motion and secondary workload stalls.
- Real workload-off baselines become possible.
- Existing prediction/slow-client safeguards survive the migration.
- Consumer memory and admission policies are inspectable.

### Negative

- Optional predictions, identity and previews may intentionally skip updates.
- Separate workers alone do not prevent shared CPU/GPU/bandwidth contention.
- Continuous moving-ball coverage needs stronger temporal annotations.
- Recorder history needs its own loss guarantees; the live stream is not archival.

## Alternatives considered

### Run all stages synchronously for each scene

This makes the primary requirement depend on secondary completion and lets
viewer/physics cost inflate location age.

### Enable the current phase scheduler as the optimization

Its many-moving policy intentionally sacrifices location detail. Preserve useful
trust/settlement ideas, but do not reuse that objective for continuous tracking.

### Create unbounded tasks or queues for every consumer

This avoids immediate admission decisions by turning old work into latency
and memory growth. Bound and coalesce replaceable work explicitly.

## Verification

- Block or overload each optional consumer and measure primary delivery/age.
- Compare location-only, identity, prediction and preview conditions on identical
  inputs with configured resource budgets.
- Dense-motion/collision clips meet completeness/gap/accuracy criteria.
- Existing prediction lifecycle and slow-client isolation tests continue to pass.
- Queue/task/memory counts stay bounded during sustained runs and shutdown.

## Revisit conditions and related records

Before accepting changed production phase policy, explicitly supersede the
conflicting perception portion of ADR-0001 and preserve its trust/prediction
protections through ADR-0002. ADR-0003 bounded public delivery remains applicable.
This proposed record does not silently change accepted production scheduling.


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 3 through 7.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/vision/phase_scheduler.py](../../../backend/vision/phase_scheduler.py).
- [Existing backend/api/routes/stream.py](../../../backend/api/routes/stream.py).

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.
