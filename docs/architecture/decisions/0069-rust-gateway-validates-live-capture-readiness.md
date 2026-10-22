# ADR-0069: Validate live capture readiness in the Rust gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

The Rust localization gateway currently accepts replay-v2 scene snapshots but
rejects the live-v3 producer handshake. Live capture already sends a
request-bound capture-session acknowledgement before reading frames. The
transitional Python gateway validates that acknowledgement against the active
revisioned configuration and verifies that delivered snapshots belong to the
acknowledged source. Leaving this boundary in Python would prevent the Rust
server from owning the production live path and would retain two different
readiness authorities.

The browser client is an established HTML/JavaScript/CSS application and is not
part of the server-language migration. Its HTTP and WebSocket response
contracts must remain compatible. The loopback producer protocol is
unauthenticated by project decision; provenance and readiness checks are not
authentication.

## Decision

Extend the Rust gateway to accept both replay-v2 and live-v3 producers. Replay
uses the existing identity handshake and accepts raw scene-v1 messages;
ADR-0076 separately extends replay-v2 to accept the versioned public-state
envelope used by paced replay. Live-v3 must include the strict capture-session
acknowledgement contract. The gateway obtains the effective capture-settings
v3 request and current
configuration revision from the existing Rust `ConfigurationAuthority`,
validates the request with the shared Rust wire parser, and validates the
acknowledgement against that exact request before returning ready. A missing
configuration path, unreadable/invalid configuration, stale revision or
digest, incompatible source report, or mismatched source-session identity
fails the handshake closed.

The server validates both the raw source snapshot and the public-state delivery
envelope. It checks scene identity, sequence and snapshot-revision ordering,
and source dimensions against the acknowledged emitted-frame geometry. The
validated raw scene is authoritative for ball locations and provenance; the
gateway rebuilds public ball fields through its own Rust lifecycle adapter.
It carries table/game metadata and elapsed localization/per-ball age durations
from the producer envelope. Producer monotonic clock values are process-local
and are never compared with the gateway's clock; the gateway rebases elapsed
durations at receipt and uses its own monotonic clock for lifecycle scheduling.
Segment durations whose clock relationship cannot be proven remain segmented.

The Rust server's HTTP and WebSocket JSON shapes remain compatible with the
existing browser client. The service remains anonymous. The Rust configuration
authority and capture report establish revision-bound readiness, not a trust
or authentication boundary.

## Consequences

### Positive

- The Rust service can receive the same live producer protocol as the current
  gateway and becomes the single production readiness validator.
- Capture settings, source report, image geometry, state provenance, and
  sequence validation share Rust-owned contracts and policy.
- Ball locations are derived from the validated raw scene in the receiving
  Rust process, while table/game metadata and producer-reported elapsed ages
  survive the process boundary.
- All server freshness decisions use a valid receiver-local monotonic clock.
- The HTML/JavaScript/CSS browser remains unchanged through this server
  migration step, and replay-v2 remains available for bounded local testing.
- No credential, token, or authentication layer is introduced.

### Negative

- The Rust gateway needs an explicit active configuration path to accept live
  producers. A replay-only development instance may still omit it.
- A live producer cannot become ready if configuration authority or source
  acknowledgement evidence is unavailable; operators receive a failed
  handshake rather than a Python fallback.
- The gateway must implement strict validation for two ingress contracts and
  keep compatibility tests against the Python receiver until cutover.
- Receiver-local freshness starts at the Rust gateway boundary. Exact camera
  exposure-to-gateway latency is available only when the source timestamp is
  comparable and verified; shared-memory publication time remains labeled as
  publication time.

## Alternatives considered

### Keep live capture ingress in Python

Rejected because Python would remain a required long-lived server process and
would continue to own live readiness policy after the Rust server migration.

### Accept a live-v3 acknowledgement without consulting current configuration

Rejected because a self-consistent producer report alone does not prove that it
matches the currently selected configuration revision and exact request.

### Trust and retain the producer's public ball payload

Rejected because it duplicates the raw scene's location data and allows a
payload to disagree with the source observation it claims to describe. The
gateway validates the producer envelope but derives client-facing ball fields
again from the validated raw scene. It carries only the table/game metadata and
elapsed age durations that the raw scene does not encode.

### Reuse producer monotonic timestamps directly

Rejected because monotonic epochs are process-local. Comparing them with the
gateway clock can create negative or fabricated ages. Only elapsed durations
from producer to handoff are carried across the boundary; gateway timestamps
are newly sampled locally.

## Verification

The gateway supports replay-v2 and live-v3 on the existing bounded loopback
scene listener. Its binary accepts an explicit `--config` path for live
capture; the reusable library config may leave that path absent to support
replay-only tests. Regression coverage verifies strict acknowledgement
validation, stale and mismatched request rejection, source identity and
geometry checks, receiver-derived ball state, preserved table/game metadata,
receiver-local freshness, producer restart epochs, and compatibility with the
existing HTTP/WebSocket fixtures. On the development Mac, `cargo test
--manifest-path perception/Cargo.toml --all-targets`, strict all-target Clippy,
formatting checks and ADR validation pass. Live hardware and Linux deployment
qualification remain separate gates.
