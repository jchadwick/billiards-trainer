# ADR-0101: Support authenticated remote perception services

- Status: accepted
- Date: 2026-10-24
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The current scene producer and gateway communicate over loopback. Users need
capture, processing, gateway, and display to run on machines selected for their
hardware while retaining bounded freshness-first delivery.

## Decision

Add independently launched camera-streamer, processor, and gateway roles for
trusted LAN or VPN deployments. Preserve the current colocated mode and loopback
transport defaults. Require encrypted, service-authenticated remote links;
keep credentials in machine-local protected settings. The gateway owns
revisioned application policy, while capture and model paths remain local to
their hardware role. Preserve acquisition identity and source timestamp
semantics across services.

## Consequences

### Positive

- Capture and inference can run on independently optimized hardware.
- Existing local deployments remain available during qualification.

### Negative

- Remote identity, credentials, reconnect behavior, and configuration agreement
  need operational support.
- Cross-host age claims require measured clock uncertainty or source clocks
  with defined synchronization.

## Alternatives considered

### Expose the current unauthenticated loopback protocol remotely

This would expand a local trust boundary without authentication or transport
confidentiality.

### Require all services to share a machine

This prevents the requested deployment on separate camera, accelerator, and
display hardware.

## Verification

- Integration tests cover cross-host operation, authentication failure,
  revision mismatch, reconnect, restart, ordering, and slow consumers.
- Deployment checks confirm loopback defaults and protected credential
  handling.
- Live audits use same-host durations unless clock offset and uncertainty are
  verified.
