# ADR-0070: Serve configuration reads from the Rust gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0066 requires every deployed server service to run in Rust while retaining
the browser as HTML/JavaScript/CSS. ADR-0065 already assigns the unified
configuration document, defaults, schema, validation and persistence to Rust.
However, the active FastAPI service still presents the public configuration
read routes. The side-by-side Rust gateway needs to take those routes one
bounded slice at a time without asking the browser to change its API calls.

The existing `GET /api/v1/config` response wraps the stored configuration in
`timestamp`, `values`, `schema_version`, `last_modified`, `is_valid` and
`validation_errors`. Its optional `section` returns only that top-level
section. `include_metadata` is accepted but ignored; stored metadata remains
present. Repeated scalar query parameters follow FastAPI's last-value-wins
lookup behavior. The `GET /api/v1/config/schema` route returns the full JSON
Schema or the selected top-level section schema directly. The Python deployment
also uses a common error envelope with an `X-Request-ID` header.

Rust authority operations are synchronous because they lock and read the
configuration file and may wait for the configured inter-process lock timeout.
Calling them directly in an async HTTP handler could block unrelated state and
WebSocket work. Rust configuration snapshots may also migrate a valid legacy
document that lacks its revision, so a snapshot is not guaranteed to be
strictly read-only.

## Decision

Add `GET /api/v1/config` and `GET /api/v1/config/schema` to the Rust gateway,
backed directly by the existing `ConfigurationAuthority`. Require an explicit
configuration path for these routes. A gateway-only process without that path
returns the established sanitized service-unavailable envelope; it does not
silently load another file or fall back to Python.

Preserve the current successful response contracts. Configuration reads
return the authority's stored document, or a single selected top-level section,
inside the existing response envelope. Both response timestamps use one
current UTC instant, matching the existing route's current behavior. Keep
`include_metadata` accepted and ignored. Preserve last-value-wins behavior for
repeated scalar query parameters. Schema reads return the raw full or section
schema with no Rust-authority protocol wrapper.

Run every authority operation through Tokio's blocking pool. Admit at most the
configured `maximum_configuration_requests` at once, defaulting to four; reject
excess concurrent operations promptly with the service-unavailable envelope.
This limits both blocking-pool work and waiters holding configuration locks.

Translate invalid sections and query validation into the deployed API's
standard error envelope, including status, timestamp, path, method, request
identifier and `X-Request-ID`. Sanitize authority/storage failures as
`SYS_002` without returning filesystem details. Do not introduce credentials,
authentication or authorization. Do not modify the browser client in this
gateway stage.

This decision covers configuration reads only. The update route is handled by
ADR-0071; reset/import, exports and download formatting, table-playing-area
adapters, and the rest of the public API remain separate gateway work. Do not
infer a mutation revision policy from these read endpoints.

## Consequences

### Positive

- Rust can serve the configuration values and schema needed by the retained
  browser without Python proxying or duplicated configuration ownership.
- The routes expose the same successful response shapes and error envelope as
  the current FastAPI service.
- Filesystem locks and revision migration cannot synchronously block the Tokio
  executor, and concurrent authority work has an explicit operational bound.
- A missing Rust authority or a damaged store fails closed and does not select
  another configuration source.

### Negative

- A gateway-only Rust process without `--config` cannot serve configuration
  reads. Live supervision and configuration routes both require the configured
  Rust authority.
- A snapshot can perform the existing legacy-revision migration and create a
  lock file; operational documentation must not characterize this endpoint as
  a purely read-only filesystem operation.
- Configuration update/reset/import/export and table-calibration routes still
  require deliberate contract migration before the Python service can be
  removed.
- The current configuration response reports retrieval time as
  `last_modified`; this implementation preserves that legacy meaning rather
  than claiming persisted modification-time provenance.

## Alternatives considered

### Keep all configuration HTTP routes in Python

Rejected as the destination because the server process would remain a deployed
Python service and the Rust authority would still depend on that server adapter.
Python remains available as a temporary side-by-side comparison during
qualification.

### Add a Python-to-Rust HTTP proxy for these reads

Rejected because the Rust service already owns the authority and can call it
directly. A proxy would add a network/process hop and preserve Python in the
server request path.

### Move all configuration endpoints in one increment

Deferred because updates, reset/import, export formatting, download behavior,
deployment-executable guards, and table-calibration validation have distinct
failure and compatibility policies. Read-route parity can be verified without
choosing a new mutation conflict policy.

### Call the configuration authority on Tokio's async executor

Rejected because authority reads acquire a filesystem lock and may wait on
disk or another process. This could delay current-state HTTP and WebSocket
work. Bounded `spawn_blocking` admission preserves isolation.

### Return authority protocol objects directly

Rejected because the Rust command protocol is an internal boundary. Returning
its `protocol_version`, `ok`, operation and wrapper fields would change the
existing browser API response.

## Verification

The Rust gateway tests exercise the real Axum router for complete and selected
configuration reads, ignored metadata flags, empty and invalid sections, full
and selected schemas, invalid and repeated query parameters, missing authority,
corrupt stores, request correlation and bounded concurrency. The full
`cargo test --manifest-path perception/Cargo.toml --all-targets` suite passes,
including the read and update router tests. Strict
`cargo clippy --manifest-path perception/Cargo.toml --all-targets -- -D warnings`
and `cargo fmt --manifest-path perception/Cargo.toml --check` pass. The Python
route fixtures and existing browser API client were unchanged for this read
slice. Passing these tests validates the implemented gateway slices; it does
not complete full route parity, deployment cutover, camera qualification or
the macOS/Linux operating gates.

## Related records

- [ADR-0065](0065-move-unified-runtime-configuration-authority-to-rust.md)
  assigns the unified configuration store and schema to Rust.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  defines the all-Rust server and standard browser boundary.
- [ADR-0068](0068-introduce-a-rust-latest-state-http-and-websocket-gateway.md)
  establishes the side-by-side Rust HTTP/WebSocket service.
- [ADR-0069](0069-rust-gateway-validates-live-capture-readiness.md)
  defines live ingress readiness from the same Rust configuration authority.
- [Rust domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks remaining gateway route and deployment work.
