# ADR-0042: Preserve live localization under bounded tracker pressure

- Status: accepted
- Date: 2026-10-21
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The real-webcam qualification process exited with
`ActiveTrackCapacityExceeded`. The live tracker retained unmatched tracks for
up to one second, matched only inside a configured image-space gate, and failed
the whole captured frame when old associations plus new detections exceeded
its fixed capacity. The capture pipeline treated that tracker error as fatal,
so an optional continuity limit stopped all future location updates. The error
did not establish whether motion, detector jitter, false detections, or a
single frame triggered the pressure. The failure and its correction are
recorded in the development-machine qualification history; the later Linux
recorded-image inference replay did not exercise a live camera or temporal
tracking pressure.

Fresh measured centers are the primary product output. Track IDs support
continuity and must remain bounded, but loss of an old association is less
harmful than loss of every current and future ball location. Scene snapshot v1
already permits a degraded outcome and a bounded reason. The detector's
configured per-frame maximum is available before camera capture begins.

## Decision

Treat `maximum_active_tracks` as a bound on retained association history, not a
reason to stop capture. After assigning current detections, when the retained
tracks and unmatched current detections exceed the bound, evict the oldest
unmatched tracks first, with track ID as a deterministic tie-breaker. Never
evict a track already matched in the current frame. Preserve monotonically
allocated track IDs and publish every eligible current detection with its
actual detector measurement. Mark a scene that evicted association history as
`degraded` with the reason `association_capacity_pressure`; consumers can then
distinguish complete locations from uninterrupted track continuity.

The live pipeline must reject a configuration before camera acquisition when
the track capacity is lower than the detector's configured maximum number of
detections. If a malformed or incompatible batch still exceeds the capacity,
retain the explicit tracking error rather than publishing an incomplete scene.
Track expiry, source-time identity, successful-empty semantics, and failed or
skipped inference behavior otherwise remain unchanged.

## Consequences

### Positive

- Association pressure no longer turns a recoverable continuity limitation
  into a capture outage.
- Every current eligible detector measurement remains present in the complete
  scene snapshot; identifiers are not recycled after eviction.
- A bounded shutdown counter reports pressure frames and evicted associations.

### Negative

- An evicted occluded or lost track cannot be reacquired with its old ID. A
  degraded scene exposes this loss of continuity, but clients must handle it.
- The retention policy is deterministic but not yet qualified on temporal
  labels for collisions, occlusion, or dense motion.

## Alternatives considered

### Increase the capacity until the failure stops

Rejected as the correctness policy because a larger bound only delays the same
failure when unmatched detections continue to accumulate. It also increases
association cost and retained state. Capacity remains configurable for measured
workloads, but localization must survive pressure at any valid bound.

### Fail the frame or process on overflow

Rejected because a tracking-history limit would suppress current measured
locations and terminate the primary capture path.

## Verification

- Rust tracker tests prove deterministic eviction, preservation of current
  measurements and matched IDs, monotonic ID allocation, degraded outcome,
  and the irreducible over-capacity frame error.
- Capture-pipeline tests prove a pressure frame is published and later frames
  continue through the same source session.
- Live-audit output reports pressure and eviction counters in its bounded
  shutdown summary.
- The live qualification report records whether pressure occurred; it does
  not treat this as evidence of temporal identity accuracy.
- Acceptance evidence: deterministic tracker tests cover oldest-unmatched
  eviction, preservation of current measurements and matched tracks,
  monotonically increasing IDs, degraded outcome, and irreducible over-capacity
  errors. Capture-pipeline tests cover publishing the pressure frame and
  continuing with subsequent frames. The Linux CPU replay did not exercise
  tracking or provide temporal identity evidence; those remain explicit
  qualification obligations.
