# ADR-0079: Reduce the first Rust-hosted client to the live view

- Status: Accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The first Mac-first runtime qualification now serves the existing HTML,
JavaScript, and CSS client from the Rust gateway and receives localized scene
state from the Rust detector over the existing WebSocket contract. The prior
client exposed dashboard, calibration, configuration, diagnostics, test-video,
and WebSocket-event surfaces. Several of those surfaces depend on Rust HTTP
routes that are not yet implemented, and none is needed to test the primary
product path: show the current camera image and fresh ball locations.

The current local fixture provides a remote MJPEG stream and the Rust detector
reads that same service independently. This proves a useful integration path,
but not frame-to-overlay synchronization, table calibration, production
camera-video delivery, ball identity, or detector accuracy against temporal
ground truth. The user has explicitly excluded ball-number classification and
has requested a working, testable system before cleanup or lower-priority
features.

## Decision

For the first Rust-hosted browser application, expose only `/live` as a
user-facing feature. Redirect `/` to `/live`; unknown or removed feature paths
show a small notice pointing to `/live`. Remove the dashboard, calibration,
configuration, diagnostics, test-video, and WebSocket-event route modules and
remove the shared feature navigation shell from the active application.

Keep the client as an ordinary browser application using HTML, JavaScript, and
CSS. Retain the existing state WebSocket, live-state model, image-coordinate
overlay, freshness display, and optional runtime-configured MJPEG viewer URL.
Do not expose calibration, settings, prediction, diagnostics, or ball-number
controls in this pass. Reusable stores, transport code, and test fixtures may
remain in source while they support `/live` or migration tests; they are not
reachable features in the application route tree.

The production browser image source remains an explicit open integration
obligation: when `api.stream.viewer_url` is unset, the current Live View falls
back to `/api/v1/stream/video`, which must be implemented by the Rust-owned
capture path before physical-camera client qualification. The remote URL used
for Mac development qualification is local runtime configuration only and is
not a production default.

## Consequences

### Positive

- The default browser route immediately tests the core image-plus-current-state
  experience without presenting unfinished feature surfaces.
- The bundle no longer imports route-level code for the removed screens, which
  reduces the browser code included in the Rust-hosted build.
- Keeping the existing state and freshness contract avoids simultaneously
  changing detector output and client message semantics.

### Negative

- Calibration and other temporarily removed tools are unavailable through the
  browser until separately restored or replaced.
- The live page still depends on some existing state-store plumbing; removing
  that unused infrastructure is deferred until it can be done without
  destabilizing the WebSocket path.
- A configured remote MJPEG image and detector state use separate stream
  connections. Their present alignment is visual smoke evidence, not a
  frame-identity synchronization guarantee.

## Alternatives considered

### Keep all existing routes and show errors for unsupported Rust endpoints

Rejected because the extra screens invite use of routes that are known to be
missing and obscure the single migration-critical path.

### Rewrite the live renderer and WebSocket client at the same time

Deferred. Reusing the existing live renderer isolates migration risk to the
route surface and allows current state freshness and coordinate behavior to
remain under their existing tests.

### Port the browser to Rust/WASM

Rejected because the user explicitly requires the browser application to stay
HTML/JavaScript/CSS. Rust owns server-side services; the browser remains a
standard web client.

## Verification

- `npm run typecheck` and the complete frontend test suite pass (70 tests).
- `npm run build` produces the static bundle served by the Rust gateway.
- Browser smoke: `/` redirects to `/live`; `/configuration` shows only the
  live-view notice; `/live` receives the remote 1280×720 image and shows a
  connected Rust WebSocket with 12 observed balls and approximately
  100–200 ms observation age during the sampled window.
- This smoke is not a production accuracy qualification: no table calibration,
  image/world coordinate ground truth, cue detection, ball-number
  classification, sustained-run latency distribution, or synchronized video
  frame identity was established.

## Related records

- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  keeps browser technology separate from server runtime ownership.
- [ADR-0077](0077-use-a-bounded-remote-mjpeg-source-for-local-rust-qualification.md)
  defines the remote MJPEG stream as a Mac-only development fixture.
- [ADR-0078](0078-serve-the-prebuilt-browser-client-from-the-rust-gateway.md)
  defines static browser delivery from the Rust gateway.
- [Mac-first live-only qualification](../../../thoughts/mac-first-migration-qualification-2026-10-22.md)
  records the measured limits of this run.
