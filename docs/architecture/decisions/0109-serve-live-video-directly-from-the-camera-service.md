# ADR-0109: Serve live video directly from the camera service

- Status: accepted
- Date: 2026-11-03
- Deciders: Project maintainers
- Supersedes: ADR-0106
- Superseded by: None

## Context

Forwarding and re-encoding live camera frames through the application
gateway adds work and makes the displayed video depend on perception.
Camera playback needs a separate transport.

## Decision

The gateway serves state and diagnostics and has no live video stream
endpoint. The browser opens the configured camera service URL in one
shared native image element. The setup API retains a bounded still for an
operator calibration action. Native video has no frame identity for
joining spatial overlays under ADR-0100.

## Consequences

### Positive

- Camera playback bypasses the API and its frame encoding and relay.
- Routes share one browser connection to the camera service.

### Negative

- The camera service URL must be reachable from the browser.
- Native playback exposes image load status rather than per-frame evidence.

## Alternatives considered

A gateway relay provides same-origin access but couples video throughput to
the API. Polling still frames would reintroduce that dependency.

## Verification

The gateway route test asserts HTTP 404 for the former streaming endpoint.
The browser receiver test verifies a direct camera URL and connection cleanup.
