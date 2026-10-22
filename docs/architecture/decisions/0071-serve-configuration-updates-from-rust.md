# ADR-0071: Serve configuration updates from the Rust gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0065 assigns configuration defaults, schema validation, revision handling
and persistence to Rust. ADR-0070 moved configuration reads to the Rust
gateway, but configuration updates still pass through FastAPI. The retained
browser sends JSON `PUT /api/v1/config` requests with optional `section`,
`validate_only` and `force_update` query parameters. FastAPI currently accepts
the route with and without its declared trailing slash through redirect
behavior. The Rust gateway must preserve the browser contract while removing
Python from this server request path.

The route also protects deployment-managed executable paths from anonymous
configuration updates. This is a field-level invariant, not user
authentication. The user has explicitly specified that the application has
no authentication anywhere. Configuration file work is synchronous and
cross-process locked; it must remain off the async executor and share the
gateway's configured admission bound with reads.

The Rust authority exposes bounded `export`, `validate` and compare-and-swap
`update` operations. `snapshot` can persist a legacy missing-revision migration,
so validation-only must use `export` and `validate`, which leave the file
unchanged. A competing process can change the document between those
operations; the original exported revision must therefore be used for the
final update.

## Decision

Serve `PUT /api/v1/config` and `PUT /api/v1/config/` directly from Rust and
preserve the current request and response fields. Keep successful updates and
validation-only results compatible with `ConfigUpdateResponse`. Keep existing
query behavior, including ignored unknown query parameters, supported boolean
spellings and last-value-wins handling for repeated scalar parameters. The
browser remains HTML, JavaScript and CSS and requires no change for this route.

Normalize an optional nonempty `section` exactly as the existing route does:
accept a matching singleton section object, otherwise wrap the request body
under that section. Reject unknown sections, empty bodies and a wrapped scalar
before authority validation. Before both validation-only and mutating flows,
compare requested changes to the current stored configuration and reject
changes to `perception.rust_live.executable_path`,
`perception.rust_live.capture_helper_executable_path` and
`perception.rust_replay.executable_path`. Repeating their current values is
allowed. Forced updates do not bypass this guard.

Serialize updates within one gateway instance with a mutation mutex, and run
the complete file-backed workflow on Tokio's blocking pool under the shared
`maximum_configuration_requests` permit. Export the baseline without defaults
or revision-migration writes, retain its revision, validate with `merge: true`,
and, when requested, update using that same baseline revision. Never replace it
with a later revision observed by validation and never retry a conflict. The
authority's file lock and revision compare-and-swap protect changes made by
other processes. An external conflict returns 409.

Validation-only returns 200 with `success` reflecting validation errors,
`updated_fields: []`, and both restart and rollback flags false. A mutating
validation error returns 422; an unforced warning returns 400; a conflict
returns 409; store, lock, and authority failures return the sanitized 503
envelope. Successful mutations return the authority's sorted changed paths,
warnings, and the historical restart flag based on the established path
prefixes. Preserve the historical `rollback_available: true` response after
successful updates without adding a rollback store. Treat the authority's
`durability: uncertain` result as committed success because the write has
already passed its atomic replacement point.

Do not add credentials, authentication or authorization middleware. Do not
change the API client or move browser code to Rust/WASM.

## Consequences

### Positive

- Rust now serves both configuration reads and updates directly from the
  Rust-owned authority; the retained browser contract remains intact.
- Validation-only works without the incidental legacy revision migration
  performed by `snapshot`.
- A per-gateway mutation gate preserves serialized local writes. The baseline
  revision and authority compare-and-swap prevent an external change from
  invalidating the executable-path check.
- Slow file operations cannot occupy the Tokio executor, and updates use the
  same bounded configuration capacity as reads.
- Executable selection remains deployment-managed while the application stays
  unauthenticated.

### Negative

- An update performs baseline export and validation before the authority's
  final validated write, adding file-lock acquisitions. This is accepted for
  contract correctness; any consolidation should follow measurement and must
  preserve the executable guard, validation-only no-write behavior and CAS.
- A caller waiting behind another local update holds one bounded configuration
  permit, so enough concurrent updates can cause reads to receive the existing
  prompt 503 overload response.
- The legacy response's `rollback_available` name overstates actual support;
  this increment preserves the field contract and adds no rollback mechanism.
- This moves only the configuration update route. Reset, import/export and
  other Python HTTP routes remain separate migration work.

## Alternatives considered

### Leave updates on the Python service

Rejected because every deployed server-side request path is required to run in
Rust, and the current Rust authority already owns validation and persistence.

### Proxy requests through Python to the Rust authority

Rejected because it retains a Python request hop and duplicates route
ownership while adding process and transport failure modes.

### Commit using the revision returned by validation

Rejected because another process could change a deployment executable path
after the initial protected-field comparison. Committing against the newer
revision would accept a value that the guard never compared with its baseline.

### Retry a revision conflict

Rejected because a retry would require re-running protected-field comparison
and validation against a new baseline and would silently change the submitted
request's concurrency semantics. Return 409 and let the client read current
state before deciding whether to submit again.

### Implement a rollback store to justify `rollback_available`

Rejected because the established route has no durable rollback transaction or
recovery contract. A new persistence mechanism would be a separate decision;
the response field is preserved only for compatibility.

## Verification

The real Axum router tests cover both URL forms, section normalization,
last-value query semantics, successful metadata-only and restart-requiring
updates, validation-only success and failure with unchanged legacy file bytes,
validation-versus-warning precedence, forced warnings, all protected executable
paths, malformed/missing/non-object bodies, bounded shared capacity, request
correlation and response envelopes. All 15 configuration HTTP integration
tests pass. The complete Rust all-target test suite passes, as do strict
all-target Clippy and Rust formatting.
The ADR validator accepts all 71 numbered records. These tests do not complete
the remaining configuration routes, Python server cutover, deployment
packaging or macOS/Linux hardware gates.

## Related records

- [ADR-0065](0065-move-unified-runtime-configuration-authority-to-rust.md)
  assigns the unified configuration authority to Rust.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  sets the Rust server and HTML/JavaScript/CSS browser boundary.
- [ADR-0070](0070-serve-configuration-reads-from-rust.md) moves configuration
  reads to the Rust gateway with the existing browser response contract.
- [Rust domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks remaining gateway route and deployment work.
