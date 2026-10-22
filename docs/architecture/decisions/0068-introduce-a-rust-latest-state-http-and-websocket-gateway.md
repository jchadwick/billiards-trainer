# ADR-0068: Introduce a Rust latest-state HTTP and WebSocket gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0066 defines the destination: every deployed server process runs in Rust,
and the browser stays a conventional HTML/JavaScript/CSS application. Python's
FastAPI service remains the active gateway during the migration. It currently
owns public HTTP and WebSocket delivery, the bounded scene receiver, startup
and shutdown, and process supervision. Replacing it before a Rust server
proves the current-state and client-delivery contracts would combine too many
failure modes in one cutover.

Rust already validates scene v1 snapshots and owns the ordered, epoch-scoped
public-state lifecycle for replay and live capture. Rust replay can publish
the existing v2 length-framed scene stream. State is latest-value data: clients
may miss intermediate updates, but a slow client must never create an old-state
backlog. The browser's current `/api/v1/game/state` and
`/api/v1/websocket/ws` state shapes are compatibility constraints. No
authentication or authorization is to be introduced. Ball-number
classification remains outside this migration.

The first server increment should make the Rust process own an executable
state-delivery path while the Python service stays available for rollback and
the rest of its routes. It must not claim to replace the complete Python
gateway or to qualify physical live capture.

## Decision

Add a separately launchable Rust server executable using Axum and Tokio in the
existing Rust package. Keep it on an alternate HTTP port for side-by-side
qualification; it does not replace Python's service or change frontend code.

For this first increment, the Rust server will:

- accept one loopback-only, bounded, length-framed scene v1 producer using the
  existing replay protocol v2 handshake, with a 256 KiB default message limit
  and a validated 32 MiB hard limit;
- validate scenes and reuse `SceneStateLifecycle` as the server's current-state
  owner for that received stream;
- expose anonymous liveness, readiness, current game-state, and state WebSocket
  routes that preserve the existing browser-facing Rust-scene contract;
- propagate state through a single latest-value slot, isolate each WebSocket
  client, and close a client whose send cannot meet a configured deadline;
- advance the public state timestamp for each accepted non-annotation update,
  while keeping each ball's last-update timestamp tied to its localization
  receipt and monotonic observation age;
- keep localization independent of optional game/session metadata and return
  unavailable metadata as unavailable rather than inferring or fabricating it.

This increment does not accept the live v3 capture acknowledgement, launch or
supervise capture/inference, expose the rest of the API, serve static assets,
or remove Python. Those remain later Rust server stages. Rust and Python
configuration, calibration, lifecycle, and session implementations remain
authoritative as already defined by their own ADRs. The client stays
HTML/JavaScript/CSS, and ball-number classification remains excluded.

## Consequences

### Positive

- Rust will have a real HTTP/WebSocket process path that can be checked against
  the current scene-state contracts before the broad gateway cutover.
- The Rust service consumes the existing replay wire contract and reuses the
  existing scene validator and freshness/order lifecycle.
- A slow WebSocket client can only observe the newest available state; it
  cannot delay scene acceptance or accumulate a historical queue.
- Python and the existing browser remain available while the server's route,
  state, restart, and resource behavior are qualified.
- Scene ingress allocation uses the default 256 KiB payload limit and cannot be
  configured above the 32 MiB ceiling.

### Negative

- A temporary second server executable and a Rust HTTP/WebSocket dependency set
  increase build and test time before they replace any Python responsibility.
- The first increment covers only health, current scene state, and state push;
  it is not a usable replacement for all current API routes or deployment.
- Replay v2 qualification does not prove live v3 handshake, camera supervision,
  static delivery, full WebSocket controls, or macOS/Linux production operation.
- The Rust-scene compatibility projection must be kept aligned with the
  current Python gateway until that gateway is retired.

## Alternatives considered

### Replace FastAPI in one cutover

Rejected because the current service combines many routes, process lifecycle,
scene ingress, supervision, WebSockets, and static assets. A one-step rewrite
would make state correctness, parity, and operational failures difficult to
localize or roll back.

### Keep a Python process proxying every request to Rust

Rejected as a destination because it leaves a deployed server-side Python
service, contrary to ADR-0066. Python may remain a temporary side-by-side
reference during qualification, but it is not the final server boundary.

### Add the gateway dependencies to the browser or use Rust/WASM

Rejected because the gateway is server-side and the user explicitly requires
the browser to remain standard HTML/JavaScript/CSS.

### Use a queue that retains every state for every client

Rejected because state is replaceable, the current state is the only state
needed for localization display, and queued old frames violate the freshness
requirement. The Tokio watch channel retains only the most recent value and
notifies subscribers when it changes ([Tokio watch documentation](https://docs.rs/tokio/latest/tokio/sync/watch/)).

## Verification

- Rust tests exercise CLI bounds, loopback-only ingress, v2 handshake, framed
  scene validation, sequence/identity restart behavior, and bounded read/write
  handling.
- HTTP and WebSocket tests compare the emitted Rust-scene shape with the
  existing v1 fixture and verify no-state readiness behavior.
- A slow-client test proves current-state publishing continues and each
  subscriber sees only a newest value or is disconnected after its configured
  send deadline; the real loopback stress passed three focused runs.
- `cargo fmt --manifest-path perception/Cargo.toml --check`,
  `cargo test --manifest-path perception/Cargo.toml --all-targets` (186 library
  tests plus all remaining targets) pass.
- `cargo clippy --manifest-path perception/Cargo.toml --all-targets -- -D warnings`
  passes.
- A separate `billiards-server` process accepted the recorded scene-v1 fixture
  over replay-v2 TCP, returned its localized state over HTTP, and shut down
  cleanly on SIGTERM. The smoke does not exercise a camera or detector.
- Live v3, full-route parity, deployment, and platform qualification remain
  explicit follow-up gates.

## Related records

- [ADR-0024](0024-scene-http-works-without-core-and-exposes-table-metadata-only-when-calibration-matches.md)
  defines the scene response's independence from Core and calibration-aligned
  table metadata.
- [ADR-0026](0026-explicit-scene-evidence-in-websocket-state.md) defines the
  anonymous WebSocket scene evidence contract.
- [ADR-0052](0052-latest-state-best-effort-freshness.md) requires
  latest-value, best-effort behavior across the state pipeline.
- [ADR-0061](0061-use-rust-public-state-lifecycle-for-paced-replay.md) defines
  the Rust replay lifecycle and source semantics.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  defines the final server/browser boundary.
- [Rust domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks the remaining gateway and domain migration stages.
