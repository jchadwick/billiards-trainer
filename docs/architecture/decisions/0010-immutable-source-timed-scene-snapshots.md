# ADR-0010: Immutable source-timed scene snapshots

- Status: accepted
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The current pipeline can poll one cached vision result repeatedly, assign fresh
Core timestamps and publish multiple apparent updates. A direct gate diagnostic
accepted a ten-second-old source under a half-second age limit when relative
age and recent poll time masked its real age. Detection failure can also become
an empty successful list. Client receipt timestamps and track IDs have acquired
meanings unrelated to original observations.

The small ball collection must survive intentional latest-state coalescing,
source restarts and asynchronous identity updates without losing provenance.

## Decision

Use immutable, versioned, complete scene snapshots as the perception boundary.
Distinguish source session/frame identity, ball observation identity, snapshot
revision and transport sequence. Repeated delivery or an identity-only update
does not create a new ball observation.

Carry original source/publication timestamps, time kind/domain, processing
completion and config/rectification/calibration revisions. Each ball retains
its last actual observation time, visibility, raw measurement, separately
identified filtered/extrapolated estimates, location uncertainty or explicit
unavailability, stable track ID and nullable number with independent confidence.

Represent successful empty observations, inference failure, skipped work,
occlusion, invalid calibration and expired estimates distinctly. Preserve
invalidity and quality reasons through transport adapters. A failed detector
must not create fresh negative evidence that balls disappeared.

Compute age at consumption from the immutable timestamp, not latest polling
or receipt. Sensor exposure time is included only when actually provided.
Specify and verify comparable Rust/Python monotonic clocks on each platform;
never serialize std::time::Instant as an assumed common epoch. Use segmented
durations when comparability is unavailable. Browser receipt clocks stay local.

At the public-state compatibility boundary, a health failure republishes only
the last known positions as explicitly cached/non-observed, preserves each
ball's source observation timestamp, and reports the failure separately. A
configured source-silence deadline may emit one stale-state update without
inventing a source timestamp. No state is considered fresh because it was
recently re-published or received. The silence deadline is required input to
the worker and must be owned by runtime configuration before production
activation.

## Consequences

### Positive

- Makes observation freshness and unique-observation accounting meaningful.
- Full snapshots recover after intentional skips or reconnect without delta replay.
- Keeps accurate-at-observation and estimated-current positions distinguishable.
- Allows identity and other components to update without falsifying ball age.
- Lets existing clients see detector failure or source silence while retaining
  the last useful map as visibly stale.

### Negative

- More explicit fields require schema evolution and compatibility adapters.
- Complete snapshots repeat a small amount of data.
- Existing clients need semantic fixes even when public routes remain stable.
- Uncertainty can remain unavailable until validated rather than being invented.
- The compatibility stream can contain a stale copy of prior locations; clients
  must honor the explicit freshness status and per-ball observation metadata.

## Alternatives considered

### Replicate DetectionResult across the process boundary

That object includes Python-specific processing, phase and mutable structures.
Copying it would preserve internal coupling rather than define a durable contract.

### Deltas and timestamp-on-publication

Deltas complicate recovery from coalescing. Publication time cannot establish
how recently the underlying ball was measured.

### One confidence and one timestamp for the whole scene

Identity confidence is not spatial uncertainty, and independently updated
components have different ages. A shared number would conceal those differences.

## Verification

- Shared fixtures cover repeated delivery, independent component updates,
  identity-only updates, out-of-order messages and restarted frame counters.
- Deterministic clocks show age increasing while source observations stop.
- Injected detector errors are distinguishable from successful empty tables.
- End-to-end adapters preserve source times, invalidity and estimate status.
- Health failures and configured source silence preserve old per-ball times,
  mark cached locations non-observed, and never advance trails as measurements.
- Source silence is measured from the latest successful localization's local
  monotonic ingress receipt; annotations and health updates do not reset it.
- Clock checks prove cross-process comparability or results stay segmented.

## Revisit conditions and related records

Extend the versioned schema when measured consumers require new evidence;
preserve old field meanings. Consider deltas only after measuring full-snapshot
cost and defining resynchronization. This record strengthens ADR-0006 and
ADR-0008 provenance without claiming their original implementation is sufficient.
Coordinates are governed by ADR-0013, identity by ADR-0019.


## Evidence and implementation tracking

Status is accepted for the v1 scene-snapshot semantics; Rust implementation and
cross-runtime qualification are not complete. The Python reference path now
retains each track's last matched wall-clock observation through tracker predictions, scheduler cache reuse,
Vision-to-Core serialization and Core `BallState.last_update`. Tracker expiry
continues to use a separate monotonic timestamp. Camera-side values remain
publication/arrival times, not exposure times. This record governs migration
stages 1 and all adapters.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/vision/models.py](../../../backend/vision/models.py).
- [Existing backend/integration_service_conversion_helpers.py](../../../backend/integration_service_conversion_helpers.py).
- [Tracker timestamp regressions](../../../backend/tests/unit/test_vision_tracker_elapsed_time.py).
- [Cached Vision timestamp regressions](../../../backend/tests/unit/test_vision_phase_integration.py).
- [Vision-to-Core timestamp regressions](../../../backend/tests/unit/test_integration_coordinate_conversion.py).
- [Scene snapshot v1 schema and contract explanation](../../../perception/contracts/README.md) and [JSON Schema](../../../perception/contracts/v1/scene-snapshot.schema.json).
- [Successful, failed, and identity-annotation fixtures](../../../perception/contracts/v1/).
- [Schema and semantic contract tests](../../../backend/tests/unit/test_perception_scene_contract.py).
- [Rust snapshot types and validation](../../../perception/src/lib.rs) and [cross-runtime fixture tests](../../../perception/tests/scene_snapshot_v1.rs).

On 2026-10-20, the focused timestamp suite passed 41 tests and the full backend
suite passed 1,162 tests with one model-dependent replay skipped. The configured
coverage gate remains below target (47.42% versus 80%) and is still a migration
completion requirement. Ruff, Black, isort, mypy and ADR validation passed after
formatting the final test change.

The v1 complete-snapshot shape is now committed in a Draft 2020-12 schema.
Fixtures and tests distinguish success with no balls from inference failure,
preserve a ball's source observation across a later identity annotation,
separate cue status, prevent table coordinates without valid calibration,
require evidence for numbered identity, and reject unsafe timestamp-number
encoding. Nineteen Python schema tests and twelve Rust decode/validate/round-trip
tests pass. Timestamp nanoseconds are decimal strings to avoid loss in IEEE-754
clients; clock-domain equality alone is not treated as proof of cross-process
comparability. This acceptance commits to the contract semantics and their Rust
representation, not a socket transport, camera runtime, or live qualification.

The Rust DTO uses Serde with `serde_json`; nullable keys are explicitly required
on decode, unknown fields fail closed, and the public encode/decode functions
run cross-field validation before data crosses the boundary. `Cargo.lock`
records the resolved versions. This keeps JSON as the inspectable v1 format
without weakening the required-field and unknown-field rules in the schema.

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.

### Failure-versus-empty implementation evidence (2026-10-20)

The retained Python reference path now carries detector outcomes across the
Vision, Core, and IntegrationService boundaries. The detector distinguishes a
successful empty table from an unavailable model and from inference, decoding,
or classification failure; failures expose only a bounded stage category.
Vision preserves the last valid ball map and its source observation times for
failed or unavailable outcomes. Core keeps its current authoritative ball
states for those outcomes, while a valid observed-empty result clears the map.
IntegrationService associates the current observation status with the matching
synchronously emitted state event. This implements ADR-0010's requirement that
failure must not become fresh negative evidence; it does not establish Rust
runtime behavior or production qualification.

The status handoff currently depends on the sole in-repository Core state-update
producer and synchronous event emission. If another producer or asynchronous
event path is added, the Core event contract needs an explicit correlation
token before this association remains safe.

The focused detector, Vision, Core, and IntegrationService regression coverage
is in [detector outcome tests](../../../backend/tests/unit/test_yolo_detector_device.py),
[Vision propagation tests](../../../backend/tests/unit/test_vision_phase_integration.py),
[Core retention tests](../../../backend/tests/unit/test_core_coordinate_invariants.py),
and [IntegrationService provenance tests](../../../backend/tests/unit/test_integration_observation_freshness.py).
The corresponding implementation is in [the detector](../../../backend/vision/detection/yolo_detector.py),
[Vision processing](../../../backend/vision/__init__.py),
[Core state updates](../../../backend/core/game_state.py), and
[IntegrationService](../../../backend/integration_service.py).

On 2026-10-20, 90 focused cross-layer tests passed. Scoped Ruff, Black, isort,
mypy, and diff checks passed; the IntegrationService CRLF line endings were
preserved and checked. The complete backend suite passed 1,258 tests with one
model-dependent replay skipped. Coverage measured 49.09% against the configured
80% threshold, so the coverage gate remains open and is not waived or lowered.
The migration and Stage 0 remain incomplete.

### Gateway consumer freshness hardening (2026-10-20)

The unwired scene-state worker now turns v1 health failures into a separate
public health state. It republishes cached ball locations with
`is_observed=false`, preserves the last measured detector confidence only as
historical evidence, and reports current position confidence as unavailable.
It does not claim a new ball observation or synthesize a source time. The
required silence timeout is measured from the latest successful localization's
local monotonic ingress receipt. Annotation and health records do not reset the
deadline; only another successful localization does. The worker emits one
stale update until a new localization arrives. FastAPI configuration and
single-publisher ownership are not yet connected, so this behavior is still a
tested prototype rather than an active production path.

The browser now accepts only explicit raw image geometry for Rust scene
states, requires declared timestamp units for Rust trail updates, resets
measured trails on engine/session changes even when track IDs repeat, rejects
rectified points from the raw-frame overlay, and retains explicit unknown
identity status. The overlays label stale measured positions as last observed,
estimated points by their estimate kind, and unavailable position confidence
as `n/a`. The selected backend adapter, worker, ingress, pipeline
replay, HTTP and WebSocket tests pass 45 tests, with one model-dependent replay
test skipped. A real-socket regression verifies that both initial locations
and later stale health state reach a subscribed client. All seven frontend test
files pass 39 tests. Scoped Ruff and Mypy pass. Repository-wide frontend
typechecking still fails in pre-existing unrelated components; no diagnostics
were reported in the changed adapter/store/type/test files. The backend
repository coverage threshold remains an overall migration gate and is not
waived.
