# ADR-0009: Rust perception engine behind the retained Python gateway

- Status: proposed
- Date: 2026-10-19
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The requested product priority is accurate, fresh ball locations, followed by
number recognition, then cue/prediction/training enrichment. The current Python
application intertwines perception, Core state and prediction admission.
The architectural review also found useful investments in the React UI,
FastAPI/WebSocket delivery, physics, calibration, model training and evaluation.

The measured detector path dominates small-payload serialization and Core costs.
A new diagnostic found CPU execution despite an MPS label; the previous run
does not establish that Python itself caused the detector cost. Rust is selected
as a proposed ownership/runtime direction, not a demonstrated speed multiplier.

## Decision

Introduce one Rust perception executable under perception/ that owns acquisition,
primary image processing, detection, association and calibrated ball observations.
Keep the Python gateway, public endpoints, frontend, physics, model training and
acceptance tooling. Connect them through the small observation/control/preview
interfaces in ADR-0010, ADR-0011 and ADR-0014.

Publish validated locations before awaiting Core or physics. In Rust mode,
ingress is the single publisher of public location state; later Core events
cannot publish a conflicting second state stream. Retained game context and
predictions remain independently consumed/output data.

Initially use one executable with internal modules rather than separate services
for capture, tracking, geometry and identity. Qualify macOS and Linux separately.
The steady-state product may retain Python indefinitely outside the perception
engine; completion does not require rewriting FastAPI or physics.

## Consequences

### Positive

- Retains working UI, API, model and training investments.
- Gives primary perception one explicit owner and enables bounded frame lifetimes.
- Allows a small recorded-input vertical slice before live camera migration.
- Keeps future inference/capture adapters replaceable behind a stable boundary.

### Negative

- Two languages and a local process boundary require coordinated packaging,
  diagnostics and lifecycle handling.
- Gateway scheduling still contributes to client latency and must be measured.
- Some Python integration routes require adapters because they access concrete
  detector/camera objects today.

## Alternatives considered

### Refactor Python to the same architecture with native inference

Python can implement the same single-owner, immutable-snapshot and bounded-work
boundaries and may meet the timing requirements after backend corrections.
Architectural simplification is not exclusive to Rust. This is the credible
lower-migration-cost alternative and remains the reference/rollback candidate.

The proposed Rust choice favors compiler-checked in-process ownership and
data-race constraints, plus CPU concurrency without the Python interpreter lock,
for the long-lived perception engine. These benefits still depend on sound
interfaces around native libraries and do not prove timestamp semantics or
cross-process protocols correct. Compare equivalent architectures/backends
fairly before default cutover; do not credit a language for changes in scheduling
or accelerator selection.

### Rewrite API, physics, frontend and tooling simultaneously

This discards useful work and obscures whether perception became better.
The measured transport costs do not justify expanding the first rewrite.

### One new service for every processing stage

This adds raw-frame IPC, supervision and operational boundaries before
demonstrating a need. Introduce additional isolation only with resource or
failure evidence.

## Verification

- Shared public API and real-socket fixtures pass for Python and Rust providers.
- A real-model recorded-input slice reaches the existing headless receiver and
  frontend without retraining or a UI rewrite.
- A deliberately blocked Core/prediction consumer cannot block primary locations.
- Tests establish one camera owner and one publisher per public stream.
- macOS/Linux build and sustained evidence satisfy ADR-0018.

## Revisit conditions and related records

Revisit gateway placement if measured ingress/publication delay materially
limits the freshness budget. Revisit process boundaries when measured
contention or fault isolation requires it. This record supplements ADRs
0002, 0003 and 0005; it does not supersede their lifecycle, delivery or
single-table-owner protections.

Evidence: [architectural review](../../../thoughts/architectural-review-2026-10-19.md)
and [migration plan](../../../thoughts/rust-perception-migration-plan.md).


## Evidence and implementation tracking

Status is proposed; Rust migration implementation and qualification are planned,
not completed. This record governs migration stages 0, 1 and 4.

The API has a validated `perception.provider` setting that defaults to
`python`. Selecting `rust` with `perception.mode=recorded_replay` enters the
supervised recorded-input path, starts the scene ingress and sole Rust scene
publisher, and marks the application healthy only after a usable localization
is published. Live camera and preview remain unavailable, and predictions are
not enabled. Selecting `rust` with `perception.mode=live` continues to fail
during preflight before dependent services start; there is no silent Python
fallback. Provider selection requires a process restart, and rollback remains
the explicit selection of `python`. Replay wiring is an implementation slice,
not evidence that live Rust perception is production-ready.

- [Migration plan](../../../thoughts/rust-perception-migration-plan.md).
- [Architectural review and evidence limitations](../../../thoughts/architectural-review-2026-10-19.md).
- [Existing backend/integration_service.py](../../../backend/integration_service.py).
- [Existing backend/api/main.py](../../../backend/api/main.py).
- [Rust API lifecycle integration test](../../../backend/tests/integration/test_rust_api_lifecycle.py).

Before accepting this decision for implementation, resolve its relevant prototype
and compatibility questions and record the evidence. Acceptance records commitment;
it does not claim all verification has passed. Link implementing changes and actual
verification artifacts as they exist. Follow the [decision register](../README.md)
for status changes and supersession.
