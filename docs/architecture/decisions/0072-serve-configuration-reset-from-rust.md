# ADR-0072: Serve configuration reset from the Rust gateway

- Status: accepted
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: None
- Superseded by: None

## Context

ADR-0065 assigns configuration persistence and validation to the Rust
authority. ADR-0070 moves configuration reads to the Rust gateway and
ADR-0071 moves updates there. `POST /api/v1/config/reset` remains a Python
route. The browser sends query parameters only; it expects a success envelope
and no body containing changed fields or revision information.

The reset route has three public modes: `all`, `user` and `section`. The
historical `user` mode resets the complete configuration because there is no
separate user namespace. All and user modes both restore trusted embedded
defaults. Section mode restores only requested top-level sections and retains
every other stored section. Section order and duplicates appear in the response
even though the Rust authority deduplicates the work. Reset has no
`force_update` option and deliberately does not apply the update route's
deployment-executable guard: a reset restores trusted defaults.

The Rust authority already supports revision-checked `replace` and
`reset_section` mutations. Its file operations are synchronous, bounded and
atomic, so the HTTP adapter must use the existing bounded blocking path and
mutation gate from ADR-0071. The entire reset must commit against one fresh
baseline revision; clients do not submit revisions and the server does not
retry conflicts.

## Decision

Serve anonymous `POST /api/v1/config/reset` directly from the Rust gateway.
Keep the browser HTML/JavaScript/CSS and preserve the route's query-only
request and existing success/error envelopes. Do not add authentication or
authorization.

Require `confirm` as a boolean. Accept `reset_type` as the case-sensitive
values `all`, `user` or `section`, defaulting to `all`. Accept repeated
`sections` parameters as an ordered list; do not split comma-separated values,
discard duplicates, or normalize case. Ignore unknown query parameters.
Preserve last-value-wins semantics for duplicate scalar parameters. Query
validation happens before route logic and aggregates invalid `confirm` and
`reset_type` errors in that order.

After query validation, reject `confirm=false` before reading configuration.
For section mode only, reject the first unknown supplied section before checking
whether the list is empty. Then use the current Rust authority defaults.
Missing or empty sections in section mode return 422. Empty `sections=` is a
list containing the empty string and therefore returns the normal invalid
section 400. For all and user modes ignore any supplied sections.

Use the existing bounded `maximum_configuration_requests` permit and gateway
mutation gate. Read the current stored configuration and revision, then use
the existing `reset_section` authority operation for section mode or `replace`
with embedded defaults for all/user mode. Always send `force_update: false`.
The original exported revision is the expected revision for the mutation. Do
not retry a conflict. Keep duplicates in `sections_reset`; for all/user return
the current full list of top-level default sections. The response keeps the
requested reset type, including `user`, and returns only the historical success
object with message, timestamp, and `data`.

Map route validation to `VAL_001`, conflicts to `GAME_002`, and authority/store
failures to sanitized `SYS_002`. Preserve the existing Python representation
of structured route details in `message` with `details: null`; framework query
errors retain their structured `validation_errors`. Keep `X-Request-ID` and
request identity fields in error responses. Successful timestamps retain the
legacy naive-UTC serialization. Do not add revision, restart, changed-field,
warning, or rollback information to the success payload.

## Consequences

### Positive

- Rust now owns the HTTP adapter for reset as well as configuration reads and
  updates, with no Python proxy on this path.
- The browser sees its existing reset choices and response shape.
- Section-only reset remains isolated from unrelated values; all and user
  intentionally restore the same complete default document.
- Existing authority compare-and-swap and atomic persistence protect resets
  from overwriting an external writer.
- The no-authentication requirement remains explicit.

### Negative

- The historical `user` label remains misleading because it resets all
  sections; changing its meaning would require a separate contract decision.
- Reset cannot repair a malformed or schema-invalid stored document because
  the authority validates the store before applying a reset.
- A section reset can still fail on warnings in untouched sections, and the
  ignored `force_update` query value does not bypass that rule.
- Configuration reset parity does not finish import/export, download, table
  calibration or the broader Python gateway migration.

## Alternatives considered

### Keep reset in Python or proxy it through Python

Rejected because the server migration requires Rust-owned server routes and
the Rust configuration authority already owns the mutations.

### Give `user` mode a partial-reset interpretation

Rejected because the existing API and tests explicitly treat it as a full
reset; there is no user-only section namespace or established alternative
definition.

### Force warnings during reset

Rejected because no force parameter is part of the established reset route.
Silently applying a new override would change behavior; callers continue to
receive a validation failure when the final document has warnings.

### Retry revision conflicts

Rejected because retries could apply a reset over state the caller never read.
Return 409 and preserve the authority's optimistic concurrency contract.

## Verification

The real Axum router tests cover required and invalid query values, aggregated
query errors, confirmation ordering, all/user/section behavior, ignored
supplied sections outside section mode, repeated and empty section values,
section-only preservation, no-op revision stability, warning behavior,
trusted executable defaults, corrupt stores, request correlation and bounded
shared configuration capacity. Existing authority tests verify stale
compare-and-swap rejection; the route sends the exported revision once and
uses the same conflict mapper as updates. The complete Rust all-target suite,
strict Clippy, formatting and ADR validation pass. These checks do not
complete remaining configuration routes, deployment cutover or macOS/Linux
hardware qualification.

## Related records

- [ADR-0065](0065-move-unified-runtime-configuration-authority-to-rust.md)
  assigns the unified revisioned configuration authority to Rust.
- [ADR-0066](0066-run-server-services-in-rust-and-retain-the-standard-web-client.md)
  keeps the browser in HTML/JavaScript/CSS while moving server services to Rust.
- [ADR-0070](0070-serve-configuration-reads-from-rust.md) moves config reads to
  the Rust gateway.
- [ADR-0071](0071-serve-configuration-updates-from-rust.md) establishes the
  bounded Rust mutation workflow and revision-conflict policy.
- [Rust domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks remaining gateway routes and deployment work.
