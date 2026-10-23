# ADR-0026: Represent scene evidence in the retained WebSocket state stream

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

> Current status: the state-evidence contract remains active in the Rust
> gateway. The Python API lifecycle suite cited below was retired with the
> compatibility replay path by ADR-0094; Rust server tests cover live delivery.

## Context

Rust scene localization is published on the existing `state` WebSocket stream
so current clients can receive fresh positions without waiting for Core or
physics. The scene payload intentionally leaves color, ball number, velocity,
cue role, and pocket state unknown unless the engine has evidence. It carries
useful evidence that the legacy Python schema did not describe: track status,
measured-versus-estimated position, source frame/session/time, calibration
coordinates, and identity confidence/evidence.

The current `backend/api/websocket/schemas.py::BallData` requires a color and
does not declare those scene fields. The broadcaster validates the state
envelope and minimum position shape but does not validate nested ball objects.
As a result, the bundled frontend can consume the Rust payload, while another
client that trusts the backend schema can reject it or discard provenance.
Requiring color would force the perception layer to invent a value for an
anonymous ball, and adding a second state stream would create competing public
location authorities.

## Decision

Keep Rust locations on the existing `state` stream and extend its versioned
Pydantic wire model with optional scene evidence fields. A missing color is
valid and means unknown. Track ID remains the ball object's `id`; numbered-ball
identity remains a separate nullable `number` plus identity status, confidence
and evidence. Source observation identity and timestamp stay distinct from the
gateway publication timestamp. Image pixels and millimeter table coordinates
remain separate typed fields with explicit units. Observed, tracked, occluded,
and lost state remains explicit; absence of velocity or identity is null or
unavailable rather than zero or a guessed label.

Legacy Python state objects remain accepted by the same model. Existing fields
retain their semantics. The additive metadata is preserved by schema parsing so
clients can use it for training and diagnostics. This decision describes the
state payload contract; it does not activate Core physics, number
classification, or trajectory prediction in Rust mode.

## Alternatives considered

### Require a color for every ball

Rejected because current Rust localization does not establish color, and
mapping an unknown ball to a guessed color would misrepresent evidence.

### Create a separate Rust-only public stream

Rejected because clients would need to merge competing state sources and
reconnect/bootstrap behavior could disagree with the existing state path.

### Keep the nested schema incomplete and rely on the bundled frontend

Rejected because the backend's declared contract should represent payloads it
actually emits, including for external consumers and future training clients.

## Consequences

### Positive

- Existing state subscriptions carry fresh Rust locations with source and
  calibration provenance.
- Anonymous balls remain useful localization results without invented color,
  number, velocity, cue role, or pocket state.
- External consumers can validate the scene fields against a documented model.
- Python-only state payloads remain compatible through optional/additive fields.

### Negative

- The WebSocket schema must evolve alongside the shared scene contract.
- Consumers must handle missing identity, color, kinematics, and physical
  coordinates explicitly.
- The current broadcaster still performs only envelope/minimum-position
  validation; strict nested validation at the shared public boundary remains a
  separate compatibility change because existing Python payloads also exceed
  the legacy schema in several ways.

## Verification

- A versioned Rust scene state fixture validates as `GameStateData` and retains
  source identity/time, track status, image geometry, physical units, and
  nullable identity fields.
- Legacy Python ball objects remain valid, including integer track IDs and
  absent color.
- A configured API lifespan with a fake child producer exercises the real
  supervisor, loopback ingress, scene worker, HTTP bootstrap and anonymous
  WebSocket subscription. Existing real-socket state delivery tests preserve
  the scene payload unchanged.

## Revisit conditions and related records

Revisit when a dedicated versioned localization endpoint or a strict nested
message validator is introduced. Any change to track identity, coordinate
units, source time, or freshness semantics must remain consistent with
ADR-0010, ADR-0013, ADR-0024, and ADR-0025. ADR-0009 governs the single public
location publisher, and ADR-0019 governs asynchronous numbered-ball identity.

## Evidence and implementation tracking

This decision implements the Stage 4 public state contract in the
[migration plan](../../../thoughts/rust-perception-migration-plan.md).
Implementation is in `backend/api/websocket/schemas.py`; versioned fixtures and
tests are in `backend/tests/fixtures/contracts/v1/` and
`backend/tests/integration/test_websocket_scene_contract.py`. The configured
Rust application lifecycle is exercised by tests in
`perception/src/server.rs` and `perception/src/public_state_lifecycle.rs`. The
offline Python compatibility scene publisher remains covered by
`backend/tests/integration/test_pipeline_replay.py`.
