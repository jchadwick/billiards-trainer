# ADR-0020: Deduplicate polled source observations before state updates

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

`VisionModule.process_frame()` can return its cached latest `DetectionResult`
when no new camera frame is available. The integration loop polls at its own
configured rate and previously treated every non-empty poll as a new frame. Each
repeat updated Core, advanced state history and refreshed
`_latest_observation_received_at`, then offered the same locations to optional
trajectory work. This turned a stopped or interrupted source into apparently
fresh state and inflated frame accounting.

The source result already carries a frame number and may carry producer process
identity, publication monotonic time, and source timestamp/kind. The service also
has a process-local vision instance identity. These fields are sufficient to
recognize a repeated result without changing public API payloads. A producer
restart may reset its frame counter, so frame number alone is insufficient.

## Decision

At the integration boundary, derive a source-observation identity before
coordinate conversion or any state/freshness/trajectory side effect. Use the
vision instance identity, source producer process ID when present, source frame
number (falling back to `DetectionResult.frame_number`), and publication
monotonic timestamp and source timestamp/kind when present; fall back to
`DetectionResult.timestamp` when source time is absent. If no frame or timestamp
identity can be formed, process the result without deduplication rather than
risk suppressing a genuine observation.

Keep the last successfully admitted identity in the integration service. A
repeat returns without updating Core, observation receipt time, trajectory
admission, or unique-frame accounting, and increments a duplicate counter. A
new frame or a changed producer process identity is admitted, including a
producer that restarts with a reset frame number. Invalid coordinate results do
not become successfully admitted observations. Core update errors remain
counted as one admitted observation attempt and increment integration errors.
The service retains that source identity to avoid retrying the same cached
result, preserves the last successfully applied observation status/receipt age,
and skips trajectory admission. A failed Core update must never present old
Core locations as fresh because newer vision data arrived.

This deduplication contract is about source observations; calibration and map
revision changes do not create a new camera observation. Reprocessing an old
source frame under a changed revision must be an explicit invalidation or
recalibration action, not an implicit freshness refresh caused by polling.

## Ownership and compatibility

`IntegrationService` owns this in-process deduplication because it is the first
boundary that both knows source provenance and mutates Core. Vision continues
to own camera capture and may continue exposing the latest cached result. Core
remains a consumer of admitted observations and needs no new deduplication
contract. The externally visible state and WebSocket schemas are unchanged;
`frame_count` now means unique admitted observations, with duplicate polls
reported separately in service metrics and optional audit events.

## Alternatives considered

### Deduplicate only by frame number

Rejected because capture frame counters can reset on writer restart and may
overlap across producer instances.

### Refresh freshness on every poll

Rejected because poll time describes when the integration loop noticed a
cached result, not when the source observation was produced. It allows stale
locations to pass freshness checks after capture stops.

### Remove polling and make capture push every result

Deferred. Event-driven delivery may reduce polling overhead, but it changes
scheduling and queueing behavior and should be evaluated after the source
identity semantics are covered by deterministic tests.

### Deduplicate in Core

Rejected because Core should not need camera-specific process and publication
identity. That would couple game-state mutation to acquisition implementation.

## Consequences

### Positive

- Repeated cached results cannot advance state history or masquerade as recent
  camera observations.
- Unique source observation throughput becomes countable independently of poll
  rate.
- A source writer restart is distinguished from a repeated frame when producer
  process identity changes.
- No wire schema or frontend migration is required for this correction.

### Negative

- Legacy sources without stable frame or timestamp identity cannot be
  deduplicated and remain explicitly less observable.
- The integration service retains a small identity tuple for its lifetime.
- A legitimate reprocessing of old pixels after calibration changes must use a
  separate invalidation/reprocessing path.
- Polling remains in place and its wasted wakeups are not addressed here.

## Verification

- Replaying one result twice leaves Core update count, receipt timestamp,
  trajectory admission count and unique frame count unchanged on the second
  call, while incrementing duplicate count.
- A newly numbered source frame is accepted.
- A writer process change with a reset frame number is accepted.
- Timestamp/frame fallbacks deduplicate cached legacy results; absent identity
  does not suppress processing.
- Coordinate conversion failures do not refresh observation age or mark the
  observation admitted.
- A Core update failure counts once as an observation attempt and one error,
  leaves the last applied status/receipt time intact, and does not start
  trajectory work; a repeated cached result is suppressed.
- Existing end-to-end replay, prediction lifecycle and WebSocket contract tests
  continue to pass.

## Revisit conditions and related records

Revisit if capture becomes push-driven, if the camera adapter supplies a stable
session UUID, or if calibration changes require explicit reprocessing of an
already captured frame. Keep such reprocessing separate from new-observation
freshness. This implements the source-time and repeated-delivery requirements of
[ADR-0010](0010-immutable-source-timed-scene-snapshots.md) and supports the
unique-observation metrics in [ADR-0008](0008-opt-in-performance-audit.md).

## Evidence and implementation tracking

The decision is accepted and implemented at the current Python integration
boundary. Seven focused tests cover repeated cached results, integration-loop
accounting, Core failure behavior, new frames, writer restart with a reset
counter, legacy timestamp fallback, unidentifiable inputs, and coordinate-error
recovery. The focused prediction/replay selection passed 29 tests with one
installed-model replay skipped because `BILLIARDS_REPLAY_MODEL_PATH` was not
configured. The latest full backend suite passed 1,160 tests with one replay
skipped; its configured 80% coverage gate was unmet at 47.39%. Full backend
Ruff, Black, isort, mypy and ADR validation pass. The coverage gate remains a
migration completion obligation.
This change does not establish Rust performance benefits or resolve the
eventual process boundary.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Integration service](../../../backend/integration_service.py).
- [Observation freshness regression tests](../../../backend/tests/unit/test_integration_observation_freshness.py).
- [Prediction lifecycle tests](../../../backend/tests/unit/test_prediction_lifecycle.py).
