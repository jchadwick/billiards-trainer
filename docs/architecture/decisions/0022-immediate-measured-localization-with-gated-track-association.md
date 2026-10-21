# ADR-0022: Immediate measured localization with gated track association

- Status: proposed
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Detection-batch v1 contains complete source-frame detections, while scene
snapshot v1 requires stable track IDs and last-observation provenance. The
generic detector does not provide numbered-ball identity. The primary product
requirement is the quickest accurate measured location; filtering and identity
must not delay or rewrite that measurement. The current Python tracker uses a
Kalman filter, Hungarian assignment, and time-based expiry, but its configuration
and image-space behavior are not yet qualified as the Rust policy. There is no
held-out temporal annotation set proving track-switch or reacquisition rates.

## Decision

Prototype a source-time tracker between successful detection batches and
scene-snapshot v1. Publish every eligible ball center as an `observed`
measurement in the same input cycle. Track association may assign a stable ID,
but it must not filter the published measurement, require a minimum hit count,
or encode numbered-ball identity.

Associate detections to active tracks with one-to-one minimum-cost assignment
under an explicit configurable image-distance gate. Use a short, bounded
constant-velocity extrapolation only to estimate association cost; do not expose
that private association position as a measured or predicted location. Preserve
the most recent actual measurement and its source timestamp. Mark unmatched
tracks `occluded`, then `lost`, according to configured elapsed source time, and
expire them at a configured maximum age. A match can reacquire an unexpired
track. Emit a new track immediately for every unmatched eligible detection.

Reset active association when engine/session identity, model artifact, applied
camera profile, configuration/rectification revision, image space, image
dimensions, source-clock domain, or source timestamp stage changes. Reconfigure
by clearing active associations while preserving the track counter and snapshot
sequence. Do not reuse track IDs within one engine/session lifetime after a
spatial reset. Reject duplicate or out-of-order source observations and batches
without source time.
The scene publisher supplies the actual snapshot-publication timestamp and
clock domain; the tracker never relabels the earlier detection-batch publication
time as scene publication.
Failed and skipped inference are errors to the tracker, never successful empty
scenes. Treat cue detections separately from ball tracks; cue absence is fresh
negative evidence only for a successful batch whose model has an explicit cue
class. Number identity remains unknown.

The recorded-image prototype emits source-image measurements and attaches table
coordinates only when the complete frame passes the immutable calibration
bundle's applied context. Missing calibration leaves table coordinates
unavailable; an incompatible configured calibration marks the scene degraded
and retains image measurements. Tracking is opt-in, paced replay only, and uses
explicit CLI policy settings. It writes validated scene-snapshot v1 JSONL
beside the raw detection batches. Live camera capture, bounded scene transport,
gateway integration, and production activation remain separate gates under
ADR-0011, ADR-0015, and ADR-0018.

## Consequences

### Positive

- The first detector measurement is published without a hit-confirmation delay.
- Track IDs remain distinct from detection indices and ball numbers.
- One-to-one assignment prevents multiple detections from updating one track.
- Source identity and elapsed time control expiry; poll rate cannot age a track.
- Predicted association positions cannot be mistaken for measured locations.
- The replay can exercise source-time tracking and calibration without changing
  detection-batch v1 or the frontend contract.

### Negative

- A false positive receives a track ID immediately; detector precision and
  temporal evaluation must quantify this risk.
- Image-space gates vary with perspective and camera changes; physical-space
  association may be preferable after qualified calibration is available.
- Scene and detection JSONL sinks flush synchronously in the measured loop, so
  replay output adds consumer interference and does not qualify the future
  bounded live transport.
- Constant-velocity association can fail at collisions, occlusion, or abrupt
  motion; it is bounded and must be evaluated on such clips.
- This prototype cannot claim physical position accuracy or live readiness.

## Alternatives considered

### Delay publication until a track is confirmed

This reduces transient IDs but delays the primary observed position and hides
the detector's immediate evidence. Retain confidence and status so consumers
can assess an unconfirmed association without withholding location.

### Greedy nearest-neighbor association

It is simpler but can consume a detection for one track and leave another
unmatched even when a globally valid one-to-one assignment exists. Compare it
only if measured table workloads show lower cost without more switches.

### Filter or extrapolate the published center

This can make motion appear smoother but moves the reported location away from
the current detector measurement. Keep any future estimate separate and require
accuracy/freshness evidence before exposing it.

### Reuse the existing Python tracker unchanged

Its current settings and elapsed-time behavior provide useful reference tests,
but copying its dependencies and state assumptions would not establish that the
same policy is suitable for the source-time Rust contract.

## Verification

- Rust tests cover immediate first-frame measurements, one-to-one global
  assignment, crossing and reacquisition, duplicate/out-of-order rejection,
  session/model/geometry/clock resets, stale-time preservation, occluded/lost
  transitions, expiry, successful empty frames, failure-versus-empty, and
  separate source/publication timestamp identity.
- Scene snapshots pass the existing semantic validator and shared Python/Rust
  fixtures; source timestamps and track IDs remain intact.
- Paced replay requires explicit tracking settings and retained raw batches,
  emits one scene for each processed source frame, and records tracking/export
  time separately. Publication time is captured after tracking immediately
  before serialization.
- Calibration tests cover matching table coordinates, applied-revision
  mismatch degradation, and clearing stale physical coordinates while keeping
  image-space tracks. A precomputed transform is accepted only when its engine,
  batch sequence, complete source-frame identity/dimensions/time, image space,
  camera profile, calibration and rectification revisions, and ordered
  detection identities/measurements match the exact detector batch.
- Temporal annotations compare track switches, reacquisition, false positives,
  gaps, and center error across aiming, dense motion, collisions and occlusion.
- No production recommendation is made until held-out temporal accuracy and
  first-detection freshness meet declared thresholds.

## Revisit conditions and related records

Use a measured refinement or physical-space association only when replay shows
that it improves track continuity without delaying or reducing observed
localization. Keep identity asynchronous under ADR-0019 and preserve immutable
source time and complete snapshot semantics under ADR-0010. This proposal does
not change the retained Python runtime or frontend payload.

## Evidence and implementation tracking

This proposal governs the first Rust temporal-association prototype in Stage 3.
Implementation under this record is experimental and does not accept the policy
for production. Acceptance requires temporal annotations and cross-platform
evidence named above.

The Rust prototype emits source-image measurements from successful timed
batches, optionally projects them through the exact frame calibration, and
validates every generated snapshot. Deterministic tracker tests cover immediate
measurements, global assignment, bounded association extrapolation,
occlusion/loss/reacquisition, capacity and publication-clock failures, source
restarts, geometry/model/clock resets, calibration revision mismatch, and
failure-versus-empty. The replay CLI requires explicit bounded tracking
configuration and paced source time; JSONL output records the validated scene
and tracking/export costs separately. These checks do not include model-backed
temporal replay, held-out physical position labels, or identity-switch labels;
ADR-0022 remains proposed.

The Rust-emitted scene fixture passes Rust round-trip validation and Python's
scene-contract tests. A model-backed paced smoke over ten mixed still images
processed nine frames and emitted nine correlated batch/scene records; one
pending image was replaced. Python's scorer matched all 57 detections with no
unmatched detections and 0.03252 px maximum center error. Mean detector pipeline
time was 92.70 ms, while tracked scene processing and synchronous scene JSONL
export averaged 0.0730 ms and 0.0614 ms. All ten synthetic 30 fps source
deadlines were missed, and processed rate was 17.58 frames/s. The 24-track
initial configuration failed closed; the completed smoke used a bounded
300-track cap on a corpus that is not temporal. These results validate model
parity and replay wiring only. They do not establish tracking accuracy, camera
freshness, physical position accuracy, or live readiness. See the [scene replay
report](../../../thoughts/performance-audits/rust-scene-replay-smoke-20261020-02/report.md)
and its [cross-runtime score](../../../thoughts/performance-audits/rust-scene-replay-smoke-20261020-02/equivalence/equivalence-20261020T162306.322661Z.md).

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Detection-batch v1](../../../perception/contracts/detection-batch/v1/README.md).
- [Scene-snapshot v1](../../../perception/contracts/v1/scene-snapshot.schema.json).
- [Existing Python tracker](../../../backend/vision/tracking/tracker.py).
- [Elapsed-time tracker regressions](../../../backend/tests/unit/test_vision_tracker_elapsed_time.py).
- [Rust source-time tracker](../../../perception/src/tracking.rs) and
  [tracked-scene contract tests](../../../perception/tests/tracking_v1.rs).
- [ADR-0010](0010-immutable-source-timed-scene-snapshots.md),
  [ADR-0013](0013-calibrated-table-positions-with-image-coordinate-compatibility.md),
  [ADR-0019](0019-asynchronous-ball-identity-separate-from-location-tracks.md).

**Precomputed calibration pairing correction (2026-10-20):** adversarial
review found that the public tracker API could pair a valid calibrated result
from one frame with the detector batch from another, then publish the physical
coordinate under the latter frame's image measurement. The calibration result
now retains the acknowledged camera-profile ID, and the tracker rejects any
precomputed result whose engine/batch/source identity, dimensions, timestamp,
image space, camera/calibration/rectification revision, detection ordering,
or source measurement differs. Rejection is transactional: it does not consume
the source observation or advance track IDs. Regression coverage exercises
stale frame identity, changed detection centers, camera profile, calibration
revision, rectification revision, and image space. These checks establish
provenance consistency, not physical accuracy; held-out labeled placement
evidence remains required.
