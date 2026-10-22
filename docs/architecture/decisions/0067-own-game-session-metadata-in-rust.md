# ADR-0067: Own game-session metadata in Rust

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

Rust live and replay already own the ordered, revisioned current scene state.
The temporary Python Core is intentionally absent in Rust mode, so the existing
game reset route otherwise has no metadata authority and can return unavailable
or synthesize a Python rack. Ball locations and calibration are the product's
primary state; game mode and session bookkeeping must not become a second owner
of localization, tracking, freshness, or prediction.

The accepted destination in [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
is a Rust-only deployed server with the existing HTML/JavaScript/CSS browser
client. The staged gateway currently remains Python, and browser integration is
deferred until Rust server qualification. The game-session step must therefore
give the gateway a small Rust-owned command boundary without changing the live
frame path or constructing a competing public-state lifecycle.

The existing Rust scene read model provides positions but not independent game
session data. History, event archives, and exports are not yet available in
Rust mode; no history data can truthfully be included in a reset backup.

## Decision

Add a separate Rust-owned, versioned session record containing game type,
active player, two scores, break status, and a monotonic revision. Persist it in
a bounded JSON file using a cross-process lock, compare-and-swap reset, and
atomic replacement. A missing record reads as an initial practice session and
is not written until a reset. Store access is through the explicitly configured
store file and a one-request Rust command-line authority with bounded stdin,
stdout, file size, lock wait, and operation deadline. The gateway derives the
authority as the packaged companion beside the configured perception
executable; there is no independently configurable session executable
selector.

If the companion binary or its settings are unavailable, continue localization
with session metadata marked unavailable. This keeps the primary detector path
independent of optional game controls. Reset requests still fail closed.

Keep this store independent of capture, scene state, table calibration, ball
identity, and prediction. Reset changes only session metadata. It must not
clear, reorder, infer, or fabricate ball positions; it must not modify camera
state or calibration. The HTTP reset response therefore projects the actual
Rust scene read model captured for that request and returns `backup_created: false`.
An explicit request to discard table state is rejected and directed to the
calibration controls.

In the transitional gateway, initialize the Rust session client only in Rust
live/replay modes. Run its blocking command outside the event loop. Session
store and limit changes require a service restart. Expose session metadata
alongside the scene response without claiming that it came from Python Core;
if any metadata read fails, keep serving localization and mark Core metadata
unavailable. Mutating reset requests fail closed on authority failure or stale
revision. Python Core remains the authority only in the explicit Python
provider path. Do not migrate classification, prediction, history, events, or
exports as part of this step. Keep the browser application HTML/JavaScript/CSS
and make no frontend changes here.

The application remains unauthenticated as required. HTTP configuration
updates and imports cannot change the installed Rust live, replay, or
capture-helper executable paths; those remain deployment-managed values.

The one-shot command is a transitional adapter boundary. Once the Rust HTTP
service replaces Python, call the same Rust session-store implementation
directly from that service rather than retaining a subprocess hop.

## Consequences

### Positive

- Rust live/replay responses can report game type and session state without
  constructing Python Core or duplicating scene locations.
- A reset cannot erase the latest measured locations or calibration.
- A missing session command cannot prevent Rust camera localization from
  starting; game controls remain unavailable until the command is installed and
  configured.
- Compare-and-swap and locking prevent stale/concurrent reset commands from
  silently overwriting newer session metadata.
- The command has explicit paths and bounded input, output, storage, locking,
  and execution behavior.
- No inference, tracking, camera, or WebSocket publication path is added to the
  metadata store.

### Negative

- The Python gateway temporarily starts the Rust command for metadata reads and
  reset writes; this per-request process overhead is transitional and should
  move to in-process Rust service calls at gateway cutover.
- Rust mode does not yet provide game history, event archives, exports, or a
  real reset backup. Those operations remain unavailable rather than being
  populated from fabricated data.
- Session scores/player/break are initialized and resettable but do not become
  authoritative gameplay outcomes until a later game-state migration adds
  evidence-backed update commands.
- The returned reset `new_state` is the Rust scene response shape, which differs
  from the legacy Core response while preserving actual localization.

## Alternatives considered

### Store session metadata in the live public-state lifecycle

Rejected because game commands and capture observations have different update
rates, ownership, and persistence requirements. A reset must not advance a
scene sequence or interfere with location freshness.

### Continue to reset Python Core in Rust mode

Rejected because Core is not running in that mode. Starting it just for reset
would reintroduce a second state engine and risk fabricated or competing ball
locations.

### Let the Python gateway persist the metadata

Rejected because the required destination is a Rust server and game metadata
is domain state. The Python gateway should translate requests only.

### Add a long-running TCP listener inside the capture process

Rejected for this slice because session control should remain available and
independently restartable, with no dependency on camera readiness or optional
preview. The one-shot command is simpler to bound and test during the gateway
transition; its process overhead is temporary and will be removed at Rust
gateway cutover.

### Return a synthetic reset rack or clear the scene on reset

Rejected because the current ball map must represent measured locations. A
reset has no authority to manufacture a rack or invalidate calibration.

## Verification

- Rust tests cover missing-store defaults, metadata reset, stale revision,
  concurrent reset, malformed/oversized records, strict request fields,
  versioned responses, and output bounds.
- Python adapter tests cover strict wire validation, bounds, process timeout,
  response overflow, and configured binary/store paths.
- API tests verify session metadata overlays without changing ball positions,
  reset returns the actual scene and only resets metadata, and calibration
  mutation is rejected.
- Lifecycle tests verify the client is only attached to Rust runtime modes and
  a cyclic optional executable path cannot prevent localization startup.
- Route tests verify metadata failures preserve scene delivery, reset returns
  the latest scene after command waits, game-session settings require restart,
  and HTTP configuration cannot replace deployed server executables.
- Rust formatting/tests/Clippy, Python formatting/lint/type checks, backend
  tests, config schema validation, and ADR validation remain required before
  calling this module complete.
- No target performance, physical-camera, or frontend qualification is claimed
  by this metadata-only module.

## Related records

- [ADR-0058](0058-use-rust-lifecycle-for-live-public-state.md) assigns live
  public-state lifecycle and freshness to Rust.
- [ADR-0061](0061-use-rust-public-state-lifecycle-for-paced-replay.md) gives
  paced replay the same Rust lifecycle.
- [ADR-0064](0064-defer-rust-web-client-and-exclude-ball-number-classification.md)
  remains superseded as to browser language and is superseded by ADR-0066; the
  classifier exclusion remains active.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  defines the final server/browser boundary.
- [Rust domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks the larger game-state migration stage.
