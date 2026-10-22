# ADR-0076: Accept Rust public-state deliveries over replay-v2

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust gateway's replay-v2 ingress originally accepted only raw
`SceneSnapshotV1` JSON. The paced Rust replay runner also sends a versioned
`PublicStateDeliveryV1` envelope so elapsed localization and per-ball ages,
table metadata, and game metadata survive the process boundary. The producer
used the replay-v2 identity handshake, but the gateway attempted to decode the
envelope as a raw scene and closed the stream with an unknown-field error.

This mismatch appeared in the Mac-first integration check: three current
1280×720 frames were read from the target's user-provided MJPEG stream, the
Rust model completed inference and produced 12 detections per frame, but the
Rust gateway returned no state because it rejected the envelope. The stream
does not provide a verified camera-exposure timestamp; local capture and
replay receipt times must not be presented as exposure-to-client latency.

The public-state envelope is already versioned independently with
`delivery_schema_version`. The server can preserve both the older raw replay
payload and the paced replay envelope without changing live-v3's strict,
configuration-bound capture handshake. The validated raw snapshot remains
the authority for ball locations, and all monotonic timestamps must be rebased
at the receiving process.

## Decision

Keep the replay-v2 producer handshake and accept either of these explicitly
defined payloads after it:

1. A raw `SceneSnapshotV1`, preserving the original replay-v2 contract.
2. A strict `PublicStateDeliveryV1` envelope with the known delivery type and
   schema version 1.

For both forms, require the snapshot engine and source-session identities to
match the identities in that connection's hello. For the envelope form,
validate the source snapshot and its public-state consistency, then rebuild
ball locations through the receiver's own `SceneStateLifecycle`. Carry only
table/game metadata and elapsed localization/per-ball ages; never reuse the
producer's monotonic instants. Replay-v2 remains usable without a runtime
configuration file. Live capture continues to require live-v3 and its
request-bound configuration acknowledgement.

## Consequences

### Positive

- The existing paced Rust replay output can be exercised through the Rust
  gateway without a Python relay or an unversioned payload change.
- Raw replay-v2 clients remain compatible, while envelope validation preserves
  source identity, age, and metadata semantics.
- The server continues deriving public ball positions from the validated raw
  scene and keeps freshness decisions on its own monotonic clock.
- A replay-only local Rust server can support accurate end-to-end checks
  without weakening live-v3 readiness requirements.

### Negative

- Replay-v2 now has two payload variants, so tests and protocol documentation
  must keep both forms explicit and reject unknown envelope versions.
- The gateway performs additional bounded JSON decoding for the envelope
  variant; the existing maximum-message limit still bounds its allocation.
- Recorded replay proves detector-to-gateway behavior for those frames, not
  continuous camera-to-browser latency or physical table-position accuracy.

## Alternatives considered

### Strip the delivery envelope from Rust replay TCP

Rejected because that discards lifecycle age and optional table/game metadata
already produced by the Rust replay engine, and diverges TCP delivery from the
existing receiver contract.

### Add a new handshake protocol version for the envelope

Rejected because the envelope has its own explicit schema version, raw
replay-v2 remains unambiguous, and the existing identity hello is sufficient
for recorded replay. A new protocol version would duplicate the live-v3
handshake without adding source-readiness guarantees.

### Route replay through the Python gateway

Rejected because local qualification must exercise the Rust service and the
accepted migration target removes Python from deployed server runtime.

## Verification

- Unit and integration tests accept both raw replay snapshots and
  `PublicStateDeliveryV1`, reject wrong identity/type/schema, and confirm age
  rebasing uses receiver-local time.
- A local `billiards-server` process receives Rust model output from three
  frames captured from the remote MJPEG stream; `/api/v1/game/state` returns
  12 observed balls with fresh state when run with the explicit 250 ms stale
  budget. Under the default 100 ms budget, similar CPU runs are correctly
  marked stale at roughly 122–139 ms of localization age.
- This verification uses recorded frames from the stream and does not yet
  qualify a continuous network capture source or the complete browser app.
- The verification records the stream's missing exposure timestamp and makes
  no exposure-to-client latency claim.

## Related records

- [ADR-0052](0052-latest-state-best-effort-freshness.md) defines newest-state
  delivery and stale rejection.
- [ADR-0061](0061-use-rust-public-state-lifecycle-for-paced-replay.md) defines
  the paced replay lifecycle and delivery envelope.
- [ADR-0068](0068-introduce-a-rust-latest-state-http-and-websocket-gateway.md)
  defines the Rust replay gateway.
- [ADR-0069](0069-rust-gateway-validates-live-capture-readiness.md) continues
  to define the strict live-v3 configuration and readiness contract.
