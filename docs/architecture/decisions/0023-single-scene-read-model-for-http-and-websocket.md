# ADR-0023: Use one scene read model for HTTP bootstrap and WebSocket state

- Status: superseded
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: ADR-0024

## Context

The Rust scene compatibility publisher currently broadcasts fresh ball
locations to the retained WebSocket state stream. The HTTP
`GET /api/v1/game/state` endpoint independently reads the Python Core state.
When a Rust scene worker is attached, clients can therefore receive a fresh
WebSocket location and then bootstrap or refresh from stale Core positions.
The scene publisher already holds the last successful localization to produce
stale health updates, but that state is private and not exposed as a read model.

The migration plan keeps Python as the gateway and Core as a secondary game
engine while making current ball locations available without waiting for
physics. Rust provider activation remains gated on camera ownership and process
supervision. This decision is only about the HTTP read path when a scene
publisher is explicitly supplied.

## Decision

Treat the scene publisher's latest complete public payload as the authoritative
location read model for both WebSocket publication and HTTP bootstrap. Expose
it through a defensive-copy accessor that contains:

- the latest complete ball payload and source identity/revisions;
- the health or source-silence status currently published to WebSocket;
- the local publication time for the state message; and
- the local receipt time of the last successful localization and per-ball
  receipt times keyed by track ID. A ball receipt advances only when that
  ball's source observation identity changes; cached estimates retain their
  earlier receipt time through newer scenes, health, and source silence.

When the read model is present, `GET /api/v1/game/state` projects its positions
and provenance into a versioned additive scene response. It combines those
locations with retained Core table/game metadata. Source timestamps remain
their original ticks, clock and domain; the HTTP timestamp is gateway
publication time, and ball `last_update` is local ingress receipt time. The
legacy integer `frame_number` carries the scene transport sequence; the
original string source frame identity remains in per-ball provenance and the
response observation status. Unknown velocity, cue-ball role and pocket state
remain explicitly unavailable rather than being inferred from old Core balls
or represented as zero measurements.

When no read model is supplied, the existing Python Core response remains
unchanged. Health and source-silence views preserve the last localization and
its receipt time while marking positions stale. They never refresh location
freshness. The scene path does not mutate Core or trigger Core state events;
predictions and physics remain separate consumers. This change does not start,
supervise, or activate a Rust producer.

## Alternatives considered

### Keep HTTP bootstrap on Core and use WebSocket as the only Rust view

Rejected because clients receive contradictory current states across two
existing public paths, and reconnecting clients can regress to an older
location set.

### Write Rust locations into Core before serving HTTP

Rejected because Core state events can activate the retained publisher and
physics lifecycle, duplicate the location stream, and couple fresh-location
delivery to secondary processing. It also obscures ownership of observation
time and track identity.

### Add a second REST-only scene cache

Rejected because it would create another mutable state copy whose freshness,
health transitions and restart invalidation could diverge from WebSocket
publication. The publisher's existing accepted scene state is the natural
single read model.

### Return only Core metadata when Rust state exists

Rejected because it does not solve the stale-ball bootstrap problem. Core
remains the source for table and game metadata only in this projection.

## Consequences

### Positive

- HTTP bootstrap and WebSocket updates refer to the same localization and
  source identity.
- Health and source silence cannot extend a ball's observation time.
- Core remains untouched on the Rust location path, avoiding duplicate event
  publishers and keeping physics secondary.
- Existing Python-only responses can remain wire-compatible.

### Negative

- The API needs an additive scene response variant to state unavailable
  kinematics and preserve source/calibration provenance.
- The read model is process-local and is lost on gateway restart until the
  producer sends a new successful scene; no persistence or replay history is
  implied.
- Table and game metadata are only as current as the retained Core context.
- Cue geometry and predictions remain outside this slice and must be exposed
  with their own source/freshness contracts.

## Verification

- Python-only `GET /api/v1/game/state` remains byte-for-byte compatible with
  its existing golden response fixture.
- A Rust scene response uses the scene location even when Core contains a
  deliberately different position; millimeters remain separate from pixels,
  and source frame/session/revision metadata is exact.
- Missing velocity, cue-ball role and pocket state serialize as unavailable.
- Ingress → worker → real WebSocket and HTTP GET expose the same location,
  source identity and stale transition.
- Health/source silence preserve `last_update`; `CoreModule.update_state` and
  Core publication events are untouched; the Rust provider still fails closed.

## Revisit conditions and related records

Revisit this read model if state publication moves to another process, if the
gateway becomes multi-worker, or if the public API introduces a dedicated
versioned localization resource. Do not add a second cache without an explicit
ownership and freshness decision. ADR-0009 governs the retained gateway,
ADR-0010 the source-time scene contract, ADR-0013 coordinate units, and
ADR-0018 provider readiness and rollback.

## Evidence and implementation tracking

The decision covers Stage 4 of the
[migration plan](../../../thoughts/rust-perception-migration-plan.md). The
read model is implemented by SceneSnapshotStatePublisher.read_model in
backend/perception/scene_state_adapter.py; HTTP projection is implemented by
scene_state_to_response and /api/v1/game/state. Verification is in
backend/tests/unit/test_scene_state_adapter.py,
backend/tests/integration/test_http_game_state_contract.py, and
backend/tests/integration/test_pipeline_replay.py::test_rust_scene_and_health_freshness_reach_real_websocket_clients.
The focused selection passes 19 tests; Ruff, Black, isort and mypy pass on the
touched backend modules, and all 23 ADRs validate. The full backend suite
passes 1,378 tests with one model-dependent skip; it still exits on the
unchanged 80% coverage threshold at 50.33%. This record was superseded by
ADR-0024 after review identified that Core availability and calibration
alignment could not be assumed. Its original implementation did not enable the
Rust provider or claim live-camera qualification.
