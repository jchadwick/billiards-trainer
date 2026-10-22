# ADR-0058: Use Rust lifecycle for live public state

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0056 assigns application-domain behavior to Rust while retaining Python as
the HTTP/WebSocket and process-supervision gateway. ADR-0057 makes the Rust live
capture path selectable, but its scene snapshots were still converted into
public state and aged by Python. That left two authorities for observation
ordering, deduplication, freshness, and expiry, and made the migration boundary
depend on the Python adapter even when Rust owned the camera and inference.

The raw scene-snapshot v1 contract is already consumed by replay and audit
tools. Replacing it outright would break those consumers and couple their
migration to live state ownership. Rust and Python monotonic clocks are process
local, so comparing their absolute values would also create invalid age
measurements.

## Decision

For explicitly selected Rust live perception, the Rust capture pipeline owns
conversion from raw observations to the current public state and applies the
producer-epoch, ordering, deduplication, stale, and expiry lifecycle before
publication. It sends a versioned public-state-delivery v1 envelope containing
both that current state and the unchanged raw scene-snapshot v1. The Python
service validates both contracts, records its own ingress receipt time, caches
the current state for existing HTTP reads, and relays it through the existing
client delivery path. The live worker must not re-adapt the Rust state or
silently fall back to Python domain semantics.

The production Rust-live configuration requires delivery schema version 1 and
fails closed if the event is missing or invalid. Raw scene replay remains on
its existing Python conversion path until replay is migrated separately. No
absolute monotonic timestamp is compared across the Rust/Python process
boundary; the gateway uses its own receipt clock for local delivery freshness.

## Consequences

### Positive

- Live state has one domain owner for transformation, ordering, and freshness.
- Existing raw-scene replay, audit inputs, frontend HTTP/WebSocket contracts,
  and the retained gateway remain available during staged migration.
- The gateway has a small responsibility: validate, cache, and deliver the
  native state rather than reinterpret it.

### Negative

- Live and replay currently use different state-production paths, so parity
  must be maintained until replay conversion moves to Rust.
- The Python process still keeps a read cache and delivery timestamps; it is a
  gateway copy of the current state, not the authority for state semantics.
- Physical-camera startup on the Linux target has not yet yielded a frame, so
  this decision has deterministic contract evidence but no target live
  accuracy or freshness qualification.

## Alternatives considered

### Keep Python-owned state conversion

Keep Python responsible for converting Rust raw snapshots and applying the
public-state lifecycle. Rejected because it preserves duplicate domain
ownership and keeps the live path dependent on Python conversion semantics.

### Replace raw scene snapshot v1

Replace the existing raw scene message with only the new public-state event.
Rejected because replay, audit, and training consumers still need the raw
observation contract and should migrate independently.

## Verification

- `cargo test --lib live_public_state_delivery_contains_rust_lifecycle_and_unchanged_raw_scene`
- `cargo test --all-targets` and `cargo fmt --all -- --check`
- Python ingress, adapter, worker, supervisor, configuration, and API lifecycle
  tests, including strict rejection of raw-only messages on configured live
  ingress and proof that live native state bypasses Python adaptation.
- ADR validation and JSON-schema validation for the new delivery contract.
- Target physical-camera startup remains an explicit qualification gap until
  the source delivers frames; no performance or accuracy claim is inferred
  from deterministic tests.
