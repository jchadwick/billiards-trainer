# ADR-0064: Defer the Rust web client and exclude ball-number classification

- Status: superseded
- Date: 2026-10-22
- Deciders: Project maintainers
- Supersedes: ADR-0056
- Superseded by: ADR-0066

## Context

The user clarified the migration boundary: the Python backend service may
remain for now, but other application runtime behavior must move to Rust. The
browser web client should be the final migration stage and then be converted to
Rust. Ball-number classification currently does not work and must not be
migrated. The product priority remains accurate, fresh ball locations; game
state and prediction features follow that primary path.

ADR-0056 correctly made the Python service a gateway and moved domain
ownership to Rust, but it explicitly retained the React frontend permanently
and included numbered-ball identity as a Rust secondary capability. Those two
scope choices no longer match the user's direction. Existing endpoint and
WebSocket contracts, the current UI during runtime migration, model assets,
training tools, and Python evaluation tools remain useful during the staged
transition.

## Decision

Keep the existing Python backend service as the HTTP/WebSocket, static-asset,
process-supervision, request/response adaptation, and operational-health
gateway. Move runtime application-domain behavior to Rust, including capture,
perception, localization/tracking, calibration/configuration authority,
game state, physics, and prediction. The gateway may translate versioned local
Rust messages into existing client contracts but may not make domain decisions
or silently fall back to Python implementations.

Do not migrate or implement ball-number classification as part of this
program. Rust localization may preserve nullable number fields in shared
contracts, but must leave identity unknown unless a future separately
authorized feature decision selects a working, accuracy-qualified classifier.
This exclusion does not block primary localization or its delivery.

Keep the current browser client and its HTTP/WebSocket contracts in place while
runtime domains migrate and qualify. After Rust runtime/domain migrations and
their compatibility, accuracy, freshness, and operating gates are complete,
replace the browser application with a Rust/WASM client. Preserve the existing
browser-facing API contracts during that replacement. Select a Rust/WASM UI
framework only when the final client stage is ready to begin; this decision
does not preselect one.

## Consequences

### Positive

- Runtime domain ownership has one Rust home while the existing service and
  browser client remain available through the migration.
- A known nonworking classifier cannot consume migration time or imply identity
  accuracy that the system does not provide.
- The Rust/WASM client conversion remains a deliberate, testable final stage
  after runtime contracts have stabilized.

### Negative

- The staged application temporarily runs a Python gateway and browser client
  alongside Rust runtime domains.
- The final client replacement adds a separate Rust/WASM toolchain, browser
  compatibility work, and client-state migration after backend-domain gates.
- Ball identity remains unavailable until a separate future decision and
  accuracy evidence justify implementing it.

## Alternatives considered

### Keep React as the permanent client

Rejected because the user explicitly requested converting the web client to
Rust after the remaining migration work is complete.

### Rewrite the web client before runtime domains are complete

Rejected because the user explicitly placed that work last, and changing both
the runtime ownership and client framework at once would obscure compatibility
failures.

### Port the existing ball-number classifier during Rust migration

Rejected because the user reports that classification does not work. Porting
an unqualified feature would add complexity without improving the primary
location path.

## Verification

- Runtime migration tests prove the Python service is only a gateway for Rust
  mode and that domain state is not started twice.
- Existing HTTP/WebSocket and browser contract tests remain green throughout
  runtime migration.
- Number fields remain explicitly unknown unless produced by a future accepted
  decision; no Rust classifier or identity fallback is added under this ADR.
- The final Rust/WASM client must preserve current routes, message contracts,
  freshness/order semantics, calibration workflows, and supported macOS/Linux
  development/deployment workflows, with browser build and end-to-end tests.
- The specific Rust/WASM framework is deferred until final-client work begins.

## Related records

- [ADR-0056](0056-retain-the-python-api-gateway-while-migrating-domain-logic-to-rust.md)
  remains historical evidence for the gateway/domain boundary and is
  superseded by this record for the complete migration scope.
- [Native domain migration plan](../../../thoughts/rust-domain-migration-plan.md)
  tracks the ordered runtime work and final Rust/WASM client stage.
