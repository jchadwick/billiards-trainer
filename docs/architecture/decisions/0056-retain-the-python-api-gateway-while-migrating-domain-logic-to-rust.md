# ADR-0056: Retain the Python API gateway while migrating application logic to Rust

- Status: superseded
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0009, ADR-0014 (domain ownership only)
- Superseded by: ADR-0064 (complete migration scope)

## Context

The user clarified that the backend service may remain Python for now, while
everything else must migrate. The earlier proposal retained the Python gateway
and application logic indefinitely, which does not meet that expectation. Rust
perception and replay integration exist, but normal live operation still uses
Python. Current evidence attributes the dominant latency to inference; this
decision does not promise a language-driven speedup.

## Decision

Keep the existing Python HTTP/WebSocket service as the client-facing gateway
during and after this migration. It may own HTTP routing, WebSocket connections,
static frontend delivery and process supervision. It must not retain camera
capture, image processing/inference, tracking/localization, configuration or
calibration authority, game/physics/prediction logic, or another application
domain implementation after those modules migrate. Rust owns those functions
and exposes versioned local interfaces to the Python gateway. Retain the React
frontend, trained weights, endpoint contracts and external Python development
tools. Authentication remains absent; macOS and Linux remain supported.

Replace domain modules incrementally while retaining the working gateway until
native lifecycle, accuracy and freshness acceptance passes. Start with complete
scene-to-public-state conversion and latest-state lifecycle in Rust, then
configuration/calibration ownership, then sole-owner capture and perception,
followed by optional physics/prediction. The Python gateway may adapt existing
client contracts over a versioned local IPC interface, but must not duplicate
domain decisions or silently fall back to Python implementations. The Rust
engine uses bounded latest-work scheduling, isolated optional consumers and the
supervised camera helper for driver failure containment. Final-hardware
performance remains a separate gate.

This replaces ADR-0009's permanent Python ownership of the perception gateway
and ADR-0014's permanent Python configuration/calibration authority. Preserve
their immutable snapshots, single publisher, revision acknowledgments,
compatible endpoints and reversible rollout contracts.

## Consequences

### Positive

- Camera and application-domain behavior have one native Rust owner.
- Existing API routes and frontend can remain available during migration.
- A narrow Python gateway avoids forcing an unrelated HTTP/WebSocket rewrite.

### Negative

- Scope includes configuration, calibration and physics replacement.
- Parallel implementations temporarily require differential verification.
- Rust binaries still need native-runtime packaging and hardware qualification.

## Alternatives considered

### Keep Python application logic behind the gateway

Rejected because perception and application-domain behavior would remain split
across runtimes and Python implementations would continue to own core decisions.

### Remove the Python gateway now

Rejected because the user permits the service to remain Python and the current
client API is useful. Keep the gateway narrow and migrate its domain work.

### Rewrite the frontend and training tools too

Rejected because these retained assets do not need to change to migrate runtime
application behavior.

## Verification

- Rust/public-state fixture outputs match the Python reference for coordinates,
  identity, cue, calibration, unknown numbers and stale/failure semantics.
- Existing frontend and API/WebSocket contracts pass through the Python gateway
  to the Rust-owned domain engine.
- Configuration/calibration tests cover validation, persistence, revisions,
  restart and exclusive camera ownership.
- Verify Python service modules contain only gateway responsibilities and no
  Python camera, inference, tracking, calibration, configuration authority or
  physics runtime is started.
- Exercise required routes, real capture and predictions through the gateway,
  then inspect the process/dependency tree to verify one Rust domain owner.
- Optional workloads cannot stall localization; Mac/Linux builds, real-model
  equivalence, physical/temporal accuracy, freshness and bounded-memory gates
  pass.
- Track outstanding work in the
  [native deployment plan](../../../thoughts/rust-domain-migration-plan.md).
