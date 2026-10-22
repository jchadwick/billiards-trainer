# ADR-0061: Use the Rust public-state lifecycle for paced replay

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0058 makes Rust the owner of current public-state conversion, observation
ordering, deduplication, freshness, and expiry for the Rust live path. Rust
recorded replay already produces and tracks the same raw `SceneSnapshotV1`, but
the Python replay gateway still adapts those snapshots and applies its own
state semantics. This leaves two deployed domain implementations and makes
replay results differ by the selected source even when their detector
observations agree.

Raw `SceneSnapshotV1` JSONL is also an established evidence and training
contract. It must remain stable while current-state delivery migrates.
Unpaced replay is used to measure capacity and can naturally exceed real-time
freshness budgets; it must not present late output as a fresh live state.

## Decision

For explicitly selected, real-time-paced Rust replay that publishes scene
messages over the Python gateway connection, apply the existing Rust
`SceneStateLifecycle` and emit the existing versioned
`PublicStateDeliveryV1` envelope. The envelope contains the lifecycle's current
public state and the exact raw source snapshot used to produce it. The Python
gateway validates the envelope, caches and relays its native state without
calling the Python scene-state adapter.

Keep raw `SceneSnapshotV1` JSONL export unchanged. Do not wrap it or advance its
schema as a side effect of changing the TCP public-state delivery payload. The
native delivery version is an explicit replay setting and must be required for
the TCP delivery path; unsupported or missing selections fail before replay
starts. Replay ingress may accept the native envelope without a live
capture-settings acknowledgement, but it must still validate the replay
producer/session identity and existing raw/envelope contracts. Live ingress
continues to require its request-bound capture acknowledgement.

Measure replay frame arrival with the replay process's local monotonic clock
before inference, then compare it to local delivery time in that same clock
domain. Keep the same versioned freshness policy as the live lifecycle. If
inference exceeds that policy, replay sends an explicit stale/health transition
and the API must not advertise old coordinates as current. Paced TCP replay is
the current-state path; unpaced capacity analysis continues to use raw batches
or JSONL and must report processing capacity separately from real-time
freshness.

## Consequences

### Positive

- Live and paced replay now share one Rust owner and one contract for current
  public state and freshness.
- The gateway remains a transport/cache layer and cannot reinterpret replay
  observations differently from live observations.
- Existing raw scene fixtures, JSONL audits, and training consumers remain
  compatible.

### Negative

- A slow detector can produce stale transitions or prevent a fresh replay
  localization; the gateway must expose that outcome rather than disguising
  it as a current result.
- Ingress configuration must distinguish live capture acknowledgement from
  replay producer identity without weakening either boundary.
- Python replay adaptation remains as a differential oracle until shared
  model-backed comparisons pass; it is no longer the selected Rust replay
  delivery authority.

## Alternatives considered

### Keep replay's public-state adapter in Python

Rejected because live and replay would continue to have separate application-
domain semantics, contrary to ADR-0056 and ADR-0058.

### Wrap raw JSONL snapshots in the new envelope

Rejected because it would break existing raw snapshot consumers and conflate
archive evidence with the latest-state delivery contract.

### Treat completion time as replay arrival time

Rejected because this would hide inference and tracking delay and could mark
late results fresh. Replay-local arrival must be recorded before processing.

### Give unpaced capacity replay a more permissive freshness policy

Deferred. A separate policy could make offline analysis convenient but risks
confusing capacity output with current-state freshness. Keep capacity results
in raw analysis formats until a concrete user-facing unpaced state consumer
requires another contract.

## Verification

- Rust tests prove TCP delivery wraps the exact raw snapshot, applies the
  versioned lifecycle, and leaves raw JSONL unchanged.
- Python tests prove native replay delivery is accepted without live capture
  acknowledgement, strict live ingress still requires its acknowledgement,
  and the replay worker bypasses Python adaptation.
- Integration tests exercise public HTTP/WebSocket state through Rust replay
  and verify stale replay outcomes never advertise old localization as current.
- `cargo fmt --all -- --check`, `cargo test --all-targets`, backend coverage,
  formatting, lint, typing, and ADR validation pass.
- Real-model replay freshness remains tied to measured runtime and the existing
  stale threshold; target physical-camera and accuracy qualification remain
  open.
