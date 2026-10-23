# ADR-0090: Keep browser client focused on live detection

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The current browser product requirement is to show the live table stream and
the latest detected state. The application routes to `/live`; the Rust server
delivers the prebuilt HTML, JavaScript, and CSS, publishes state over its
WebSocket contract, and the browser displays the camera's direct MJPEG URL.
Predicted trajectories and other client features are secondary to current
ball locations.

The web source still contained configuration, monitoring, diagnostics,
navigation, persistence, general video controls, and trajectory overlays from
an earlier broader UI. These features were not reachable from the current
route graph. The app entry point nevertheless constructed the old aggregate
store, which imported configuration, game, vision, UI, and persistence state
into the live page. The Playwright diagnostic targeted a route that did not
exist and did not make behavior assertions. The starter README also described
proxy and developer-tool behavior no longer provided by the app.

Keeping unreachable feature trees and a generic state architecture increases
the number of contracts that can diverge from the Rust server while making
errors in direct video and fresh-state display harder to isolate. The cleanup
must preserve the actual `/live` behavior, anonymous connectivity, direct
stream URL semantics, freshness rejection, and coordinate alignment.

## Decision

Keep the browser as a small HTML/JavaScript/CSS client whose supported product
surface is `/live` plus the root redirect. Its client state is limited to the
reconnecting state WebSocket, observable connection status, and latest
accepted game state. Reject out-of-order state updates and reset ordering
identity when the producer session changes. Continue to load the camera image
directly from the configured MJPEG URL; do not route camera video through the
Rust API or a JavaScript video proxy.

Use a small application stylesheet for this single page rather than retaining
a utility-CSS framework, its unused theme configuration, or a separate CSS
post-processing stack. Keep the visual rules in ordinary CSS beside the
browser source.

Remove configuration, diagnostics, monitoring, generic navigation,
persistence, unused video controls and overlays, unconnected accessibility
utilities, stale API and store models, and tests that only validated those
unreachable surfaces. Keep focused tests for latest-state freshness,
producer-session ordering, the fixed 16-ball grid, image/overlay coordinates,
direct MJPEG URL handling, and WebSocket reconnect and delivery behavior. Keep
Vitest, typechecking, and the Vite build as the maintained local checks.

Do not add a feature back solely because its source or prior tests exist. A
future client capability must have an owned product use case, a current Rust
server contract, a route or explicit live-view interaction, and focused tests.
Real-browser E2E coverage for `/live` against a running Rust state stream and
MJPEG source remains a follow-up because the current unit tests use controlled
inputs.

## Consequences

### Positive

- The live page no longer initializes unrelated application stores or imports
  unreachable feature modules.
- The published client and its tests describe the current direct-stream and
  latest-state contracts instead of a dormant broader application.
- Removing unused UI and dependency surfaces reduces bundle inputs and
  maintenance work while keeping current image, state, and coordinate behavior.
- Removing the unused Tailwind/PostCSS pipeline also removes its vulnerable
  build-time dependency chain; production and all-dependency npm audits are
  clean after the supported dependency updates.

### Negative

- Configuration, monitoring, diagnostics, trajectory display, and generic
  video controls are no longer available in the browser. Each requires a new
  product decision and implementation before it can return.
- Existing component-level tests for removed screens no longer provide
  evidence; current browser E2E coverage is still missing.

## Alternatives considered

### Keep dormant features in the client but leave their routes hidden

Rejected because they still pulled unused stores and dependencies into the
runtime and preserved stale API contracts without a current user path.

### Replace the current UI with WebAssembly or a Rust frontend

Rejected because the user explicitly requires the browser application to
remain HTML/JavaScript/CSS. Rust remains the server-side runtime.

### Remove all browser tests and rely on the Rust gateway suite

Rejected because Rust tests cannot verify client-side freshness handling,
direct MJPEG selection, coordinate mapping, or rendering assumptions. Keep the
focused frontend suite and add real-browser coverage when an owned integration
fixture is available.

## Verification

- Vitest passed with 7 files and 16 tests after removing unrelated feature
  tests.
- TypeScript typecheck passed.
- Production Vite build passed and produced assets for Rust static delivery.
- npm reported zero vulnerabilities in production and development dependency
  trees after removing Tailwind/PostCSS and updating the test runner.
- Rust and Python suites, shell syntax, and ADR structure are verified by the
  repository-wide check after this decision is recorded.
