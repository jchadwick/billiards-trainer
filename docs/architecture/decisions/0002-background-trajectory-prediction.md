# ADR-0002: Background trajectory prediction

- Status: accepted
- Date: 2026-10-18
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The phase-aware vision scheduler can provide a valid aiming observation while
multiball physics is still calculating the preceding one. Running the complete
calculation inside the integration coroutine delays later frame processing and
can make a result obsolete before it is broadcast. A calculation also reads
mutable state assembled from vision and Core, so moving it to a worker requires
an explicit ownership and lifecycle boundary.

## Decision

Run multiball cue-shot prediction in a supervised background task and offload
the synchronous physics call to a worker thread. Keep at most one calculation
active and one pending request; replace pending work with the newest eligible
observation. Build each request from detached copies of cue, ball, table, and
settings inputs so the worker never reads live vision or Core state.

Before publishing, require the service to remain active and the current
observation to remain eligible in the same phase, map revision, and calibration
revision. Reject results whose source frame or latest observation exceeds the
configured `vision.processing.phase_scheduler.maximum_observation_age_seconds`
limit. Keep calculator access serialized. On stop, clear pending work and wait
for the active worker to finish; do not cancel an asyncio wrapper around a
thread that cannot be stopped.

Preserve ADR-0001's lifecycle: motion freezes the exact last valid live
prediction, motion and recovery do not create or replace live predictions, and
calibration changes invalidate the saved spatial reference. An obsolete
background result is discarded before it can affect this state.

## Consequences

### Positive

- Physics calculations no longer block integration from processing new vision
  observations.
- Overload is bounded to one active calculation and the newest pending
  observation.
- Results from changed or stale observations cannot overwrite a valid frozen
  comparison.
- Worker inputs have a clear ownership boundary, and shutdown waits for thread
  completion without pretending that task cancellation stops the thread.

### Negative

- A calculation already running during shutdown must finish before service stop
  completes.
- Prediction results may be dropped when their phase, map, calibration, or
  freshness context changes before publication.

## Alternatives considered

### Run physics inside the integration coroutine

This blocks frame processing for the duration of each calculation and increases
the chance that the result no longer describes the current aiming context.

### Cancel the calculation task during shutdown

Cancelling an asyncio task awaiting a worker thread does not stop the thread.
It can leave calculator work running after the service has reported shutdown.

## Verification

- Deterministic lifecycle tests block a physics worker with synchronization
  events and verify event-loop responsiveness and newest-pending coalescing.
- Tests reject results after phase, map, calibration, or observation-age
  changes.
- Shutdown tests verify the service drains active physics work without
  publishing its result after the lifecycle ends.
- Input snapshot tests verify later mutations do not change captured physics
  state.
