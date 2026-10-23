# ADR-0091: Retire disconnected projector test and browser packaging tools

- Status: accepted
- Date: 2026-10-23
- Deciders: Project maintainers
- Supersedes: ADR-0086's retention of projector test and browser-packaging tooling
- Superseded by: None

## Context

The native projector receives a custom UDP protocol on loopback, but the Rust
product server does not publish that protocol. The retained Python senders
generated fabricated trajectories and calibration messages; their unit tests
checked those same fabricated payloads, not capture, detection, or delivery
from the product. A shell helper launched the disconnected sender and used a
broad process-name kill. The optional love.js packaging path built a separate
browser artifact without the product's HTML/JavaScript/CSS client, WebSocket,
or state contract.

The projector application remains a distinct native client with a possible
future projection use case. Its Lua coordinate-contract tests validate a
meaningful boundary independently of fabricated network traffic. The primary
browser client remains `frontend/web`.

## Decision

Remove the Python UDP sender tests, the combined local test launcher, and the
love.js browser-packaging workflow and its documentation. Keep the native LÖVE
application, local run helper, and hardware-free Lua coordinate-contract test.
Do not replace the removed fake transport with Rust code: it would only test a
protocol with no production publisher. Restore end-to-end projector testing
when a supported backend-to-projector adapter has an owned transport contract.

## Consequences

### Positive

- Projector checks no longer imply that synthetic UDP messages qualify live
  product delivery.
- The repository has one maintained browser application, the actual
  HTML/JavaScript/CSS `/live` client.
- Projector-specific test and packaging workflows no longer add Python,
  Node-based WASM packaging, or unowned fake data paths.

### Negative

- Developers can no longer preview projector trajectory rendering with
  fabricated local messages; coordinate validation remains covered, but a
  production publisher and visual integration test are future work.

## Alternatives considered

### Keep synthetic UDP tools and packaging

Rejected because the sender's payload round trip was isolated from the
product's WebSocket and latest-state contracts, while the love.js artifact was
not the supported browser application. Keeping either workflow risks treating
an unconnected demonstration as product coverage.

## Verification

- `lua frontend/projector/test_coordinate_contract.lua` passes.
- `luac -p frontend/projector/core/coordinate_contract.lua` passes.
- Repository search finds no live references to the removed sender, launcher,
  or browser-packaging workflow.
- Rust, browser, backend-compatibility, shell, and ADR checks pass after the
  retirement.
