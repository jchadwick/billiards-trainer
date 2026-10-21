# ADR-0024: Serve scene localization independently from calibration-gated Core metadata

- Status: accepted
- Date: 2026-10-20
- Deciders: Project maintainers
- Supersedes: ADR-0023
- Superseded by: None

## Context

ADR-0023 established one read model for HTTP bootstrap and WebSocket state. Its
implementation projected scene positions alongside Core table/game metadata,
but review identified two unsafe assumptions at that boundary:

1. The HTTP route still required an initialized Core module before it read the
   scene snapshot, even though localization is the primary result and the scene
   publisher is intentionally independent of physics/game logic.
2. Core table geometry was paired with Rust scene coordinates without evidence
   that both came from the same calibration generation. `TableState` currently
   has no calibration revision, and matching dimensions do not prove matching
   camera geometry.

The same review found that the publisher labeled its post-adaptation time as a
localization receipt time. The ingress already had a monotonic receipt for
freshness deadlines but did not retain a paired UTC value that could be exposed
as the public per-ball timestamp. The frontend also had only a Core-shaped
`GameStateResponse` type, which could not represent nullable Rust kinematics or
missing table metadata.

## Decision

Keep the publisher's defensive scene read model as the one source of current
Rust locations for both HTTP and WebSocket. The scene HTTP response must not
require Core to exist or be initialized. If Core is absent, return localization
with null table/game metadata, empty events, and explicit Core/table metadata
status. When no scene read model exists, preserve the existing Python-only
Core-required behavior and response fixture.

Core may contribute game type, events, and a separate Core metadata timestamp;
these values do not replace scene positions. Emit Core table geometry only
when both the scene and the Core table have known calibration revisions and
the values match exactly. If either revision is absent, omit geometry and mark
the table metadata `unverified`. If both revisions exist but differ, omit
geometry and mark it `mismatch`. If Core/table context is absent, mark it
`unavailable`. Do not infer alignment from equal dimensions, current config, or
wall-clock proximity. Propagating a trustworthy calibration revision into Core
is a separate implementation gate; until then, the Rust scene response must
not expose retained Core table geometry.

Capture a paired UTC and monotonic receipt time when a complete scene snapshot
is accepted by ingress. Forward UTC through the worker and use it for HTTP
localization timestamps. Keep monotonic receipt time for same-process freshness
deadlines and never serialize it as UTC. Gateway state publication time remains
distinct from localization receipt; health and source-silence updates advance
only publication time, not localization freshness.

Keep source timestamps in their original ticks, clock, domain and stage. The
legacy integer `frame_number` remains the transport sequence; source frame ID
and per-ball source observation remain available in provenance. Do not infer
velocity, cue-ball role, pocket state, or ball identity from stale Core state.
The frontend models the Core and scene response variants as a union, with
unknown scene fields remaining nullable and no coercion to zero or false.

Scene response validity requires a successful scene observation and
`ball_positions_status == observed`. A successful observed-empty frame is a
valid empty result. A scene that reported successful detection but could not
provide usable image positions for its tracks is invalid.

## Alternatives considered

### Continue requiring Core for the scene HTTP route

Rejected because starting, stopping, or failing the secondary game engine
would make the primary localization unavailable. This reverses the project's
localization-first ownership boundary.

### Continue returning Core table geometry as an unmarked convenience

Rejected because stale or differently calibrated table geometry can make
accurate ball coordinates appear wrong and can cause downstream clients to
derive incorrect pockets, boundaries, or training labels. An explicit
unverified status with geometry omitted is safer than plausible-looking
coordinates without provenance.

### Infer calibration alignment from equal table dimensions or latest config

Rejected because different camera poses and homographies can share dimensions,
and a current config value does not prove which revision produced retained Core
geometry. Only source-owned revision equality is evidence of alignment.

### Encode unavailable velocity and role as zero/false

Rejected because these values are valid measurements, not absence markers.
Nullable typed fields prevent clients from treating unknown physics or identity
as facts.

## Consequences

### Positive

- Current localization remains available when Core is uninitialized or absent.
- HTTP and WebSocket continue to expose the same scene snapshot, identity and
  freshness transitions.
- Core table geometry cannot silently cross a calibration boundary.
- UTC receipt timestamps represent ingress receipt rather than later adapter
  completion, while monotonic deadlines retain their correct clock semantics.
- Frontend types describe both response variants without corrupting unknown
  physics or metadata.

### Negative

- The scene response can omit table geometry until Core calibration revision
  provenance is modeled and populated. Consumers must handle null table data.
- Core game metadata can be stale relative to the scene; the separate Core
  status/timestamp makes that source distinction visible but does not reconcile
  the two state machines.
- The HTTP read model remains process-local and unavailable after gateway
  restart until a new successful scene arrives.

## Verification

- The existing Python-only HTTP golden response remains byte-for-byte stable,
  and the no-scene path still returns 503 when Core is unavailable.
- Scene HTTP returns current ball positions without Core; unknown table/game
  metadata and Core status serialize explicitly.
- Matching calibration revisions expose table geometry; missing and mismatched
  revisions omit it while retaining scene localization.
- Successful observed-empty state is valid; unusable scene geometry is invalid.
- Ingress captures paired wall-clock UTC and monotonic receipt values; the
  worker forwards UTC into per-ball and localization receipts. Monotonic values
  remain used for silence deadlines.
- Production replay keeps Core updates and prediction calls untouched, and
  HTTP/WebSocket positions, source identity and stale status agree.
- Frontend API-client and response-union tests retain null scene fields; the
  frontend build/typecheck must be rerun, with unrelated baseline type errors
  reported separately.

## Revisit conditions and related records

Revisit this boundary when Core can carry a source-owned calibration revision,
when a dedicated versioned localization endpoint is introduced, or when state
publication moves to a multi-process/multi-worker deployment. ADR-0023 records
the preceding shared-read-model decision and remains the historical rationale;
this record supersedes only its assumption that Core metadata can be paired
without an availability/alignment contract. ADR-0009 governs the retained
gateway, ADR-0010 the source-time scene contract, ADR-0013 coordinate units,
and ADR-0018 provider readiness and rollback.

## Evidence and implementation tracking

Implemented in the scene state HTTP projection, ingress receipt model, scene
worker, frontend API response types/client, and HTTP/worker integration tests.
Current focused results and the full-suite/coverage status are tracked in the
[migration plan](../../../thoughts/rust-perception-migration-plan.md). This
decision does not enable or qualify the Rust provider or claim live-camera
performance.
